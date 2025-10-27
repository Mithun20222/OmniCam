# src/server.py
import os
import cv2
import time
import uuid
import threading
import tempfile
import queue
import asyncio
import numpy as np
from datetime import datetime
from collections import deque
from deepface import DeepFace
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv

# ----------------- CONFIG -----------------
AUTHORIZED_DIR = "data/authorized"
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Camera configuration - will be populated dynamically
cameras = {}

# Shared frame buffers for each camera
camera_frames = {}  # {camera_id: {"frame": np.array, "timestamp": float}}
camera_locks = {}   # {camera_id: threading.Lock()}

def find_working_cameras():
    """Find all working cameras on the system."""
    working_cameras = {}
    
    print("🔍 Scanning for available cameras...")
    
    backends = [
        (cv2.CAP_MSMF, "Microsoft Media Foundation"),
        (cv2.CAP_ANY, "Any Available"),
        (cv2.CAP_DSHOW, "DirectShow")
    ]
    
    # Test if DirectShow works
    dshow_works = False
    try:
        test_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if test_cap.isOpened():
            ret, frame = test_cap.read()
            if ret and frame is not None:
                dshow_works = True
            test_cap.release()
    except:
        pass
    
    if not dshow_works:
        print("⚠️ DirectShow not working on this system, skipping...")
        backends = [b for b in backends if b[0] != cv2.CAP_DSHOW]
    
    # Only test the first working backend to avoid conflicts
    for backend, backend_name in backends:
        print(f"Testing {backend_name} backend...")
        found_any = False
        
        for i in range(5):
            if i in working_cameras:  # Skip if already found with another backend
                continue
                
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✅ Found working camera {i} with {backend_name}")
                        working_cameras[i] = {
                            "id": f"cam{i+1}",
                            "location": f"Camera {i+1}",
                            "backend": backend,
                            "backend_name": backend_name
                        }
                        found_any = True
                    cap.release()
                else:
                    cap.release()
            except Exception as e:
                print(f"⚠️ Error testing camera {i} with {backend_name}: {e}")
                if 'cap' in locals():
                    cap.release()
        
        # Stop after first working backend to avoid conflicts
        if found_any:
            print(f"✅ Using {backend_name} backend for all cameras")
            break
    
    if not working_cameras:
        print("❌ No working cameras found!")
        working_cameras[0] = {
            "id": "cam1", 
            "location": "Default Camera",
            "backend": cv2.CAP_ANY,
            "backend_name": "Default"
        }
        print("⚠️ Using fallback camera configuration")
    
    return working_cameras

# Initialize cameras
cameras = find_working_cameras()

# Initialize frame buffers and locks for each camera
for cam_id in cameras.keys():
    camera_frames[cam_id] = {"frame": None, "timestamp": 0}
    camera_locks[cam_id] = threading.Lock()

# ----------------- Globals -----------------
flags = {}
intruder_embeddings = {}
intruder_count = 0
events = deque(maxlen=500)
recent_labels = deque(maxlen=5)
last_seen = {}
RESET_TIME = 3600
ws_clients = []
last_snapshot_time = {}
SNAPSHOT_COOLDOWN = 5

# WebSocket event queue
event_queue = queue.Queue()

# ----------------- Supabase -----------------
load_dotenv()
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

def save_intruder(intruder_id, emb, face_crop, camera_id="cam_0"):
    """Upload intruder image + metadata to Supabase"""
    tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp_file.name, face_crop)
    file_name = f"{intruder_id}_{uuid.uuid4().hex}.jpg"

    try:
        with open(tmp_file.name, "rb") as f:
            supabase.storage.from_("intruder-photos").upload(
                file_name,
                f,
                {"content-type": "image/jpeg", "x-upsert": "true"}
            )

        photo_url = supabase.storage.from_("intruder-photos").get_public_url(file_name)
        data = {
            "intruder_id": intruder_id,
            "camera_id": camera_id,
            "embedding": emb.tolist(),
            "photo_url": photo_url,
        }
        supabase.table("intruders").insert(data).execute()
        print(f"☁️ Logged intruder {intruder_id} with photo → {photo_url}")
    except Exception as e:
        print("❌ Upload failed → Check Supabase policies for bucket 'intruder-photos'")
        print("Error details:", e)
    finally:
        try:
            os.unlink(tmp_file.name)
        except:
            pass

# ----------------- FACE UTILS -----------------
def build_embeddings():
    embeddings = {}
    for person in os.listdir(AUTHORIZED_DIR):
        person_dir = os.path.join(AUTHORIZED_DIR, person)
        if not os.path.isdir(person_dir):
            continue
        reps = []
        for img in os.listdir(person_dir):
            try:
                rep = DeepFace.represent(
                    img_path=os.path.join(person_dir, img),
                    model_name="SFace",
                    detector_backend="opencv",
                    enforce_detection=False,
                )[0]["embedding"]
                reps.append(rep)
            except Exception as e:
                print(f"Error processing {img}: {e}")
        if reps:
            embeddings[person] = reps
            flags[person] = 0
    print(f"✅ Loaded {len(embeddings)} authorized identities.")
    return embeddings

def cosine_distance(a, b):
    a, b = np.array(a), np.array(b)
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize_face(frame, known_embeddings, threshold=0.6):
    try:
        emb = DeepFace.represent(
            frame,
            model_name="SFace",
            detector_backend="opencv",
            enforce_detection=False,
        )[0]["embedding"]

        best_match, best_score = None, 1e6
        for person, reps in known_embeddings.items():
            for ref_emb in reps:
                dist = cosine_distance(emb, ref_emb)
                if dist < best_score:
                    best_score, best_match = dist, person
        return (best_match, emb) if best_score < threshold else (None, emb)
    except Exception as e:
        print("Recognition error:", e)
    return None, None

# ----------------- SNAPSHOTS + EVENTS -----------------
def save_snapshot(frame, label, x, y, w, h, color):
    annotated = frame.copy()
    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
    cv2.putText(annotated, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    filename = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(SNAPSHOT_DIR, filename)
    cv2.imwrite(path, annotated)
    return path

def broadcast_event(event):
    """Queue event for WebSocket broadcasting"""
    event_queue.put(event)

def log_event_no_snapshot(camera_id, label):
    """Log event without snapshot (for authorized users)."""
    event = {
        "id": uuid.uuid4().hex,
        "camera_id": cameras[camera_id]["id"],
        "location": cameras[camera_id]["location"],
        "timestamp": datetime.utcnow().isoformat(),
        "label": label,
        "image_path": None
    }
    events.append(event)
    broadcast_event(event)
    return event

def log_event(camera_id, label, frame, x, y, w, h, color):
    """Log confirmed intruder event with snapshot."""
    current_time = time.time()
    should_snapshot = True
    
    if "Intruder" in label and "intruder_" in label:
        intruder_id = label.split("(")[1].split(")")[0]
        if intruder_id in last_snapshot_time and (current_time - last_snapshot_time[intruder_id]) < SNAPSHOT_COOLDOWN:
            should_snapshot = False
        else:
            last_snapshot_time[intruder_id] = current_time
    
    event = {
        "id": uuid.uuid4().hex,
        "camera_id": cameras[camera_id]["id"],
        "location": cameras[camera_id]["location"],
        "timestamp": datetime.utcnow().isoformat(),
        "label": label,
        "image_path": save_snapshot(frame, label, x, y, w, h, color) if should_snapshot else None
    }
    events.append(event)
    broadcast_event(event)
    return event

# ----------------- WEBSOCKET BROADCASTER -----------------
def websocket_broadcaster():
    """Background thread to broadcast events to WebSocket clients"""
    print("🔌 WebSocket broadcaster started")
    while True:
        try:
            event = event_queue.get(timeout=1)
            dead_clients = []
            
            for ws in ws_clients[:]:
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(ws.send_json(event))
                    loop.close()
                except Exception as e:
                    dead_clients.append(ws)
            
            for ws in dead_clients:
                if ws in ws_clients:
                    ws_clients.remove(ws)
                    
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Broadcaster error: {e}")

# ----------------- CAMERA CAPTURE THREAD -----------------
def camera_capture_thread(camera_id):
    """
    Dedicated thread to continuously capture frames from camera
    and store them in shared buffer. This runs independently.
    """
    camera_info = cameras.get(camera_id, {})
    backend = camera_info.get("backend", cv2.CAP_ANY)
    backend_name = camera_info.get("backend_name", "Default")
    
    print(f"📹 Starting capture thread for camera {camera_id} with {backend_name}")
    
    cap = cv2.VideoCapture(camera_id, backend)
    if not cap.isOpened():
        print(f"❌ Failed to open camera {camera_id}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print(f"✅ Camera {camera_id} capture started")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Failed to read from camera {camera_id}")
            time.sleep(0.1)
            continue
        
        # Update shared frame buffer
        with camera_locks[camera_id]:
            camera_frames[camera_id]["frame"] = frame.copy()
            camera_frames[camera_id]["timestamp"] = time.time()
    
    cap.release()

# ----------------- FACE RECOGNITION THREAD -----------------
def run_camera(camera_id, known_embeddings):
    """
    Process frames from shared buffer for face recognition.
    This reads from camera_frames instead of opening camera directly.
    """
    global intruder_count
    
    print(f"🎥 Starting face recognition for camera {camera_id}")
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    face_trackers = {}
    TRACKER_TIMEOUT = 2.0
    frame_count = 0
    PROCESS_EVERY_N_FRAMES = 3  # Process every 3rd frame

    # Wait for first frame
    while camera_frames[camera_id]["frame"] is None:
        print(f"⏳ Waiting for first frame from camera {camera_id}...")
        time.sleep(0.5)
    
    print(f"✅ Face recognition started for camera {camera_id}")

    while True:
        # Get frame from shared buffer
        with camera_locks[camera_id]:
            frame = camera_frames[camera_id]["frame"]
            if frame is None:
                time.sleep(0.1)
                continue
            frame = frame.copy()  # Copy to avoid threading issues

        frame_count += 1
        
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            time.sleep(0.033)  # ~30 FPS
            continue

        current_time = time.time()
        
        # Clean up old trackers
        face_trackers = {
            k: v for k, v in face_trackers.items() 
            if current_time - v["last_seen"] < TRACKER_TIMEOUT
        }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            if w < 80 or h < 80:
                continue
                
            face_key = f"{x//50}_{y//50}"
            
            if face_key not in face_trackers:
                face_trackers[face_key] = {
                    "buffer": [],
                    "last_seen": current_time,
                    "verified": False,
                    "label": None
                }
            
            tracker = face_trackers[face_key]
            tracker["last_seen"] = current_time
            
            if tracker["verified"] and tracker["label"]:
                continue
            
            face_crop = frame[y:y+h, x:x+w]
            recognized, emb = recognize_face(face_crop, known_embeddings)

            if recognized:
                last_seen[recognized] = current_time
                if flags.get(recognized, 0) == 0:
                    flags[recognized] = 1
                    log_event_no_snapshot(camera_id, f"Authorized: {recognized}")
                    print(f"✅ Authorized: {recognized}")
                
                tracker["verified"] = True
                tracker["label"] = f"Authorized: {recognized}"
                tracker["buffer"] = []

            else:
                if emb is None:
                    continue
                    
                matched_intruder = None
                for intruder_id, reps in intruder_embeddings.items():
                    for ref_emb in reps:
                        if cosine_distance(emb, ref_emb) < 0.5:
                            matched_intruder = intruder_id
                            break
                    if matched_intruder:
                        break

                if matched_intruder:
                    if not tracker["verified"]:
                        label = f"Intruder ({matched_intruder})"
                        log_event(camera_id, label, frame, x, y, w, h, (0, 0, 255))
                        print(f"⚠️ Known intruder: {matched_intruder}")
                        tracker["verified"] = True
                        tracker["label"] = label
                        tracker["buffer"] = []
                else:
                    tracker["buffer"].append(emb)
                    
                    if len(tracker["buffer"]) >= 8:
                        intruder_id = f"intruder_{intruder_count}"
                        intruder_embeddings[intruder_id] = [emb]
                        flags[intruder_id] = -1
                        
                        threading.Thread(
                            target=save_intruder,
                            args=(intruder_id, np.array(emb), face_crop, cameras[camera_id]["id"]),
                            daemon=True
                        ).start()
                        
                        intruder_count += 1
                        label = f"Intruder ({intruder_id})"
                        log_event(camera_id, label, frame, x, y, w, h, (0, 0, 255))
                        print(f"🚨 NEW INTRUDER: {intruder_id}")
                        
                        tracker["verified"] = True
                        tracker["label"] = label
                        tracker["buffer"] = []
                    else:
                        verification_data = {
                            "type": "verification",
                            "camera_id": cameras[camera_id]["id"],
                            "location": cameras[camera_id]["location"],
                            "status": "verifying",
                            "buffer_count": len(tracker["buffer"])
                        }
                        broadcast_event(verification_data)

# ----------------- FASTAPI APP -----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def gen_frames(camera_id=0):
    """
    Stream frames from shared buffer instead of opening camera directly.
    This prevents conflicts with the capture thread.
    """
    print(f"📺 Starting stream for camera {camera_id}")
    
    # Wait for camera to start capturing
    timeout = 10
    start_time = time.time()
    while camera_frames.get(camera_id, {}).get("frame") is None:
        if time.time() - start_time > timeout:
            print(f"❌ Timeout waiting for camera {camera_id}")
            
            import io
            from PIL import Image, ImageDraw
            
            img = Image.new('RGB', (640, 480), color=(50, 50, 50))
            draw = ImageDraw.Draw(img)
            draw.text((200, 220), "Camera Unavailable", fill=(255, 255, 255))
            
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            while True:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + img_byte_arr + b'\r\n')
                time.sleep(1)
            return
        
        time.sleep(0.1)
    
    print(f"✅ Streaming camera {camera_id}")
    
    while True:
        with camera_locks[camera_id]:
            frame = camera_frames[camera_id]["frame"]
            if frame is None:
                continue
            frame = frame.copy()
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

@app.get("/camera_feed/{camera_id}")
def camera_feed(camera_id: int):
    if camera_id < 1 or camera_id > 10:
        raise HTTPException(status_code=400, detail="Invalid camera ID")
    
    actual_camera_id = camera_id - 1
    
    if actual_camera_id not in cameras:
        if cameras:
            actual_camera_id = list(cameras.keys())[0]
        else:
            actual_camera_id = 0
    
    return StreamingResponse(
        gen_frames(actual_camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
    )

@app.on_event("startup")
def startup_event():
    known_embeddings = build_embeddings()
    
    print(f"🚀 Starting OmniCam with {len(cameras)} detected camera(s)")
    
    # Start WebSocket broadcaster
    threading.Thread(target=websocket_broadcaster, daemon=True).start()
    
    # Start camera capture and processing threads
    for cam_id, cam_info in cameras.items():
        print(f"🎥 Starting camera {cam_id} ({cam_info['id']}) - {cam_info['location']}")
        
        # Thread 1: Capture frames continuously
        threading.Thread(
            target=camera_capture_thread, 
            args=(cam_id,), 
            daemon=True
        ).start()
        
        # Thread 2: Process frames for face recognition
        threading.Thread(
            target=run_camera, 
            args=(cam_id, known_embeddings), 
            daemon=True
        ).start()

@app.get("/cameras")
def get_cameras():
    return list(cameras.values())

@app.get("/cameras/refresh")
def refresh_cameras():
    global cameras
    cameras = find_working_cameras()
    return {
        "message": f"Refreshed cameras. Found {len(cameras)} working camera(s).",
        "cameras": list(cameras.values())
    }

@app.get("/events")
def get_events():
    return list(events)[-50:]

@app.get("/events/latest")
def latest_event():
    return list(events)[-1] if events else {}

@app.get("/snapshot/{event_id}")
def get_snapshot(event_id: str):
    for e in events:
        if e["id"] == event_id and e.get("image_path"):
            return FileResponse(e["image_path"])
    return {"error": "Snapshot not found"}

@app.websocket("/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.append(websocket)
    print(f"✅ WebSocket client connected. Total: {len(ws_clients)}")
    try:
        while True:
            await websocket.receive_text()
    except Exception as e:
        print(f"WebSocket disconnected: {e}")
    finally:
        if websocket in ws_clients:
            ws_clients.remove(websocket)
        print(f"❌ WebSocket disconnected. Total: {len(ws_clients)}")

@app.get("/flags")
def get_flags():
    return flags

@app.get("/stats")
def get_stats():
    return {
        "total_events": len(events),
        "intruder_count": intruder_count,
        "authorized_count": len([f for f in flags if flags[f] >= 0]),
        "active_cameras": len(cameras),
        "ws_clients": len(ws_clients)
    }