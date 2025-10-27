# src/server.py
import os
import cv2
import time
import uuid
import threading
import tempfile
import numpy as np
from datetime import datetime
from collections import deque
from deepface import DeepFace
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv

# ----------------- CONFIG -----------------
AUTHORIZED_DIR = "data/authorized"
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Camera configuration - will be populated dynamically
cameras = {}

def find_working_cameras():
    """Find all working cameras on the system."""
    working_cameras = {}
    
    print("Scanning for available cameras...")
    
    # Try different backends in order of preference, but skip DirectShow if it fails
    backends = [
        (cv2.CAP_MSMF, "Microsoft Media Foundation"),
        (cv2.CAP_ANY, "Any Available"),
        (cv2.CAP_DSHOW, "DirectShow")  # Try DirectShow last
    ]
    
    # First, test if DirectShow works at all
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
        print("DirectShow not working on this system, skipping...")
        backends = [b for b in backends if b[0] != cv2.CAP_DSHOW]
    
    for backend, backend_name in backends:
        print(f"Testing {backend_name} backend...")
        
        # Test up to 5 camera indices
        for i in range(5):
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    # Try to read a frame to confirm it's working
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"Found working camera {i} with {backend_name}")
                        working_cameras[i] = {
                            "id": f"cam{i+1}",
                            "location": f"Camera {i+1}",
                            "backend": backend,
                            "backend_name": backend_name
                        }
                    cap.release()
                else:
                    cap.release()
            except Exception as e:
                print(f"Error testing camera {i} with {backend_name}: {e}")
                if 'cap' in locals():
                    cap.release()
    
    if not working_cameras:
        print("No working cameras found!")
        # Fallback to default camera 0 with ANY backend
        working_cameras[0] = {
            "id": "cam1", 
            "location": "Default Camera",
            "backend": cv2.CAP_ANY,
            "backend_name": "Default"
        }
        print("Using fallback camera configuration")
    
    return working_cameras

# Initialize cameras
cameras = find_working_cameras()

# ----------------- Globals -----------------
flags = {}
intruder_embeddings = {}
intruder_count = 0
intruder_buffer = []
events = []
recent_labels = deque(maxlen=5)
last_seen = {}
RESET_TIME = 3600  # 1 hour
ws_clients = []
# Track last snapshot time to prevent excessive snapshots
last_snapshot_time = {}
SNAPSHOT_COOLDOWN = 5  # seconds between snapshots for same person

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
        print(f"Logged intruder {intruder_id} with photo → {photo_url}")

    except Exception as e:
        print("Upload failed → Check Supabase policies for bucket 'intruder-photos'")
        print("Error details:", e)


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
    print(f"Loaded {len(embeddings)} authorized identities.")
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


def log_event_no_snapshot(camera_id, label, frame, x, y, w, h, color):
    """Log event without snapshot (for authorized users)."""
    event = {
        "id": uuid.uuid4().hex,
        "camera_id": cameras[camera_id]["id"],
        "location": cameras[camera_id]["location"],
        "timestamp": datetime.utcnow().isoformat(),
        "label": label,
        "image_path": None  # No snapshot for authorized users
    }
    
    # Add to events
    events.append(event)
    
    # push via WebSocket
    for ws in ws_clients:
        try:
            import asyncio
            asyncio.run(ws.send_json(event))
        except:
            pass
    
    return event


def log_event(camera_id, label, frame, x, y, w, h, color):
    """Log confirmed intruder event with snapshot. Only called for intruders."""
    current_time = time.time()
    
    # Only take snapshot for intruders
    should_snapshot = True
    
    if "Intruder" in label and "intruder_" in label:
        intruder_id = label.split("(")[1].split(")")[0]
        if intruder_id in last_snapshot_time and (current_time - last_snapshot_time[intruder_id]) < SNAPSHOT_COOLDOWN:
            should_snapshot = False
        else:
            last_snapshot_time[intruder_id] = current_time
    
    # Create event with snapshot (only for intruders)
    event = {
        "id": uuid.uuid4().hex,
        "camera_id": cameras[camera_id]["id"],
        "location": cameras[camera_id]["location"],
        "timestamp": datetime.utcnow().isoformat(),
        "label": label,
        "image_path": save_snapshot(frame, label, x, y, w, h, color) if should_snapshot else None
    }
    
    # Add to events
    events.append(event)
    
    # push via WebSocket
    for ws in ws_clients:
        try:
            import asyncio
            asyncio.run(ws.send_json(event))
        except:
            pass
    
    return event


# ----------------- CAMERA LOOP -----------------
def run_camera(camera_id, known_embeddings):
    global intruder_count, intruder_buffer
    
    # Get the backend for this camera
    camera_info = cameras.get(camera_id, {})
    backend = camera_info.get("backend", cv2.CAP_ANY)
    backend_name = camera_info.get("backend_name", "Default")
    
    print(f"🎥 Starting camera {camera_id} with {backend_name} backend...")
    
    cap = cv2.VideoCapture(camera_id, backend)
    if not cap.isOpened():
        print(f"Failed to open camera {camera_id} with {backend_name}")
        return
    
    print(f"Camera {camera_id} opened successfully with {backend_name}")
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            if w < 80 or h < 80:
                continue
            face_crop = frame[y:y+h, x:x+w]
            recognized, emb = recognize_face(face_crop, known_embeddings)

            if recognized:  # Authorized
                intruder_buffer.clear()
                last_seen[recognized] = time.time()
                if flags[recognized] == 0:
                    flags[recognized] = 1
                    # Log authorized user detection without snapshot
                    log_event_no_snapshot(camera_id, f"Authorized: {recognized}", frame, x, y, w, h, (0, 255, 0))
                label, color = f"Authorized: {recognized}", (0, 255, 0)

            else:  # Intruder check
                matched_intruder = None
                for intruder_id, reps in intruder_embeddings.items():
                    for ref_emb in reps:
                        if cosine_distance(emb, ref_emb) < 0.5:
                            matched_intruder = intruder_id
                            break
                    if matched_intruder:
                        break

                if matched_intruder:
                    label, color = f"Intruder ({matched_intruder})", (0, 0, 255)
                    intruder_buffer.clear()
                else:
                    intruder_buffer.append(emb)
                    if len(intruder_buffer) >= 8:  # only after 8 frames
                        intruder_id = f"intruder_{intruder_count}"
                        intruder_embeddings[intruder_id] = [emb]
                        flags[intruder_id] = -1
                        save_intruder(intruder_id, np.array(emb), face_crop)
                        intruder_count += 1
                        label, color = f"Intruder ({intruder_id})", (0, 0, 255)
                        intruder_buffer.clear()
                        print(f"⚠️ Intruder detected: {intruder_id}")
                        # Only log event when intruder is confirmed
                        log_event(camera_id, label, frame, x, y, w, h, color)
                    else:
                        label, color = "Verifying...", (0, 255, 255)
                        # Send verification status via WebSocket
                        verification_data = {
                            "type": "verification",
                            "camera_id": cameras[camera_id]["id"],
                            "location": cameras[camera_id]["location"],
                            "status": "verifying"
                        }
                        for ws in ws_clients:
                            try:
                                import asyncio
                                asyncio.run(ws.send_json(verification_data))
                            except:
                                pass

            recent_labels.append(label)
            stable_label = max(set(recent_labels), key=recent_labels.count)

    cap.release()


# ----------------- FASTAPI APP -----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Replace the gen_frames and camera_feed functions with these fixed versions:

from fastapi.responses import StreamingResponse

def gen_frames(camera_id=0):
    """Generate frames for camera stream."""
    # Get the backend for this camera
    camera_info = cameras.get(camera_id, {})
    backend = camera_info.get("backend", cv2.CAP_ANY)
    backend_name = camera_info.get("backend_name", "Default")
    
    print(f"Starting stream for camera {camera_id} with {backend_name} backend...")
    
    cap = cv2.VideoCapture(int(camera_id), backend)
    if not cap.isOpened():
        print(f"Failed to open camera {camera_id} for streaming with {backend_name}")
        # Return a blank frame or error image instead of returning nothing
        import io
        from PIL import Image, ImageDraw, ImageFont
        
        # Create error image
        img = Image.new('RGB', (640, 480), color=(50, 50, 50))
        draw = ImageDraw.Draw(img)
        draw.text((200, 220), "Camera Unavailable", fill=(255, 255, 255))
        draw.text((220, 250), f"Camera {camera_id}", fill=(200, 200, 200))
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Yield error frame continuously
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_byte_arr + b'\r\n')
            time.sleep(1)  # Update every second
        return
        
    print(f"Camera {camera_id} stream started successfully with {backend_name}")
    
    # Set camera properties for streaming
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print(f"Failed to read frame from camera {camera_id}")
                time.sleep(0.1)
                continue
                
            # Encode frame as JPEG with quality 80
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except Exception as e:
        print(f"Error in gen_frames for camera {camera_id}: {e}")
    finally:
        cap.release()
        print(f"Stream ended for camera {camera_id}")


@app.get("/camera_feed/{camera_id}")
def camera_feed(camera_id: int):
    """
    Serve camera feed. 
    Note: camera_id from URL is 1-based (cam1, cam2), 
    but our cameras dict uses 0-based indexing.
    """
    # Convert 1-based URL parameter to 0-based camera index
    actual_camera_id = camera_id - 1
    
    print(f"Camera feed requested for camera {camera_id} (actual index: {actual_camera_id})")
    
    # Verify camera exists
    if actual_camera_id not in cameras:
        print(f"Camera {actual_camera_id} not found in cameras dict")
        print(f"Available cameras: {list(cameras.keys())}")
        # Return the first available camera as fallback
        if cameras:
            actual_camera_id = list(cameras.keys())[0]
            print(f"Using fallback camera {actual_camera_id}")
        else:
            actual_camera_id = 0
            print(f"No cameras available, using default camera 0")
    
    return StreamingResponse(
        gen_frames(actual_camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.on_event("startup")
def startup_event():
    known_embeddings = build_embeddings()
    
    print(f"Starting OmniCam with {len(cameras)} detected camera(s)")
    
    # Start camera threads for all detected cameras
    for cam_id, cam_info in cameras.items():
        print(f"🎥 Starting camera {cam_id} ({cam_info['id']}) - {cam_info['location']}")
        threading.Thread(
            target=run_camera, args=(cam_id, known_embeddings), daemon=True
        ).start()


@app.get("/cameras")
def get_cameras():
    return list(cameras.values())

@app.get("/cameras/refresh")
def refresh_cameras():
    """Refresh camera detection and return updated camera list."""
    global cameras
    cameras = find_working_cameras()
    return {
        "message": f"Refreshed cameras. Found {len(cameras)} working camera(s).",
        "cameras": list(cameras.values())
    }

@app.get("/cameras/test/{camera_id}")
def test_camera(camera_id: int):
    """Test a specific camera to see if it's working."""
    camera_info = cameras.get(camera_id, {})
    backend = camera_info.get("backend", cv2.CAP_ANY)
    backend_name = camera_info.get("backend_name", "Default")
    
    try:
        cap = cv2.VideoCapture(camera_id, backend)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.release()
                return {
                    "success": True,
                    "message": f"Camera {camera_id} is working with {backend_name}",
                    "frame_size": f"{frame.shape[1]}x{frame.shape[0]}",
                    "backend": backend_name
                }
            else:
                cap.release()
                return {
                    "success": False,
                    "message": f"Camera {camera_id} opened but can't capture frames",
                    "backend": backend_name
                }
        else:
            return {
                "success": False,
                "message": f"Camera {camera_id} failed to open with {backend_name}",
                "backend": backend_name
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error testing camera {camera_id}: {str(e)}",
            "backend": backend_name
        }

@app.get("/camera_feed/test/{camera_id}")
def test_camera_feed(camera_id: int):
    """Test camera feed endpoint directly."""
    try:
        camera_info = cameras.get(camera_id, {})
        backend = camera_info.get("backend", cv2.CAP_ANY)
        backend_name = camera_info.get("backend_name", "Default")
        
        cap = cv2.VideoCapture(camera_id, backend)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    cap.release()
                    return {
                        "success": True,
                        "message": f"Camera {camera_id} feed test successful",
                        "frame_size": f"{frame.shape[1]}x{frame.shape[0]}",
                        "backend": backend_name,
                        "jpeg_size": len(buffer)
                    }
                else:
                    cap.release()
                    return {"success": False, "message": "Failed to encode frame as JPEG"}
            else:
                cap.release()
                return {"success": False, "message": "Failed to capture frame"}
        else:
            return {"success": False, "message": f"Failed to open camera {camera_id}"}
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}


@app.get("/events")
def get_events():
    return events[-50:]


@app.get("/events/latest")
def latest_event():
    return events[-1] if events else {}


@app.get("/snapshot/{event_id}")
def get_snapshot(event_id: str):
    for e in events:
        if e["id"] == event_id:
            if e.get("image_path"):
                return FileResponse(e["image_path"])
            else:
                return {"error": "No snapshot available for this event"}
    return {"error": "Event not found"}


@app.websocket("/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        ws_clients.remove(websocket)


@app.get("/flags")
def get_flags():
    return flags
