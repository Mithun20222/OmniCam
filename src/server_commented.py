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
import requests
import serial

# Configuration and directory setup
AUTHORIZED_DIR = "data/authorized"
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Camera configuration - populated dynamically during startup
cameras = {}

# Shared frame buffers for each camera to avoid concurrent access issues
camera_frames = {}  # Dictionary mapping camera_id to {"frame": np.array, "timestamp": float}
camera_locks = {}   # Dictionary mapping camera_id to threading.Lock()

def find_working_cameras():
    """
    Scan system for available cameras and test which ones work.
    Returns dictionary mapping camera index to camera configuration.
    """
    working_cameras = {}
    
    print("Scanning for available cameras...")
    
    # List of camera backends to test, in order of preference
    backends = [
        (cv2.CAP_MSMF, "Microsoft Media Foundation"),
        (cv2.CAP_ANY, "Any Available"),
        (cv2.CAP_DSHOW, "DirectShow")
    ]
    
    # Test if DirectShow backend is functional on this system
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
    
    # Test only the first working backend to prevent conflicts between backends
    for backend, backend_name in backends:
        print(f"Testing {backend_name} backend...")
        found_any = False
        
        # Test camera indices 0 through 4
        for i in range(5):
            if i in working_cameras:  # Skip if already found with another backend
                continue
                
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"Found working camera {i} with {backend_name}")
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
                print(f"Error testing camera {i} with {backend_name}: {e}")
                if 'cap' in locals():
                    cap.release()
        
        # Stop after first working backend to avoid conflicts
        if found_any:
            print(f"SUCCESS: Using {backend_name} backend for all cameras")
            break
    
    if not working_cameras:
        print("ERROR: No working cameras found!")
        working_cameras[0] = {
            "id": "cam1", 
            "location": "Default Camera",
            "backend": cv2.CAP_ANY,
            "backend_name": "Default"
        }
        print("Using fallback camera configuration")
    
    return working_cameras

# Initialize cameras by scanning system
cameras = find_working_cameras()

# Initialize frame buffers and locks for each detected camera
for cam_id in cameras.keys():
    camera_frames[cam_id] = {"frame": None, "timestamp": 0}
    camera_locks[cam_id] = threading.Lock()

# Global state variables
flags = {}  # Dictionary tracking detection status for each person
intruder_embeddings = {}  # Dictionary storing facial embeddings for detected intruders
intruder_count = 0  # Counter for assigning unique intruder IDs
events = deque(maxlen=500)  # Recent events queue with max 500 items
recent_labels = deque(maxlen=5)  # Queue of recent recognition labels
last_seen = {}  # Dictionary tracking last seen timestamp for each person
RESET_TIME = 3600  # Time in seconds before resetting detection flags
ws_clients = []  # List of active WebSocket connections
last_snapshot_time = {}  # Dictionary tracking last snapshot time per intruder
SNAPSHOT_COOLDOWN = 5  # Minimum seconds between snapshots for same intruder

# Queue for broadcasting events to WebSocket clients
event_queue = queue.Queue()

# Initialize Supabase client for cloud storage
load_dotenv()
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Telegram bot configuration for alerts
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Initialize ESP32 serial connection for servo control
try:
    esp = serial.Serial('COM5', 115200, timeout=1)  # Adjust COM port as needed
    print("Connected to ESP32 on COM5")
except Exception as e:
    esp = None
    print("ESP32 not connected:", e)


def save_intruder(intruder_id, emb, face_crop, camera_id="cam_0"):
    """
    Upload intruder photo and metadata to Supabase cloud storage.
    
    Args:
        intruder_id: Unique identifier for the intruder
        emb: Facial embedding array
        face_crop: Cropped face image
        camera_id: ID of camera that detected the intruder
    """
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
        print(f"CLOUD STORAGE: Logged intruder {intruder_id} with photo -> {photo_url}")
    except Exception as e:
        print("Upload failed -> Check Supabase policies for bucket 'intruder-photos'")
        print("Error details:", e)
    finally:
        try:
            os.unlink(tmp_file.name)
        except:
            pass

def send_telegram_alert(intruder_id, photo_url):
    """
    Send Telegram message with photo when intruder is detected.
    
    Args:
        intruder_id: Unique identifier for the intruder
        photo_url: Public URL of uploaded intruder photo
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("WARNING: Telegram credentials missing, skipping alert.")
        return

    message = f"ALERT: Intruder Detected!\n\nID: {intruder_id}\nPhoto: {photo_url}"

    try:
        # Send text message
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": message}
        )

        # Send image if available
        if photo_url:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                data={"chat_id": TELEGRAM_CHAT_ID, "photo": photo_url}
            )

        print(f"TELEGRAM: Alert sent for {intruder_id}")
    except Exception as e:
        print("ERROR: Telegram send failed:", e)

def send_servo_command(error_x, error_y):
    """
    Send pan/tilt correction commands to ESP32 servo controller.
    
    Args:
        error_x: Horizontal offset from center (positive = face is right of center)
        error_y: Vertical offset from center (positive = face is below center)
    """
    if esp is None:
        return

    try:
        # Send horizontal pan command if face is off-center
        if abs(error_x) > 25:
            direction_x = "RIGHT" if error_x > 0 else "LEFT"
            esp.write(f"PAN:{direction_x}\n".encode())

        # Send vertical tilt command if face is off-center
        if abs(error_y) > 25:
            direction_y = "UP" if error_y < 0 else "DOWN"
            esp.write(f"TILT:{direction_y}\n".encode())

    except Exception as e:
        print("WARNING: Serial communication error:", e)

# Face recognition utility functions
def build_embeddings():
    """
    Build facial embeddings database from authorized user photos.
    Scans AUTHORIZED_DIR for subdirectories, each containing photos of one person.
    
    Returns:
        Dictionary mapping person name to list of facial embeddings
    """
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
    """
    Calculate cosine distance between two vectors.
    Returns value between 0 (identical) and 2 (opposite).
    
    Args:
        a: First vector
        b: Second vector
    
    Returns:
        Cosine distance (1 - cosine similarity)
    """
    a, b = np.array(a), np.array(b)
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize_face(frame, known_embeddings, threshold=0.6):
    """
    Recognize a face in the given frame by comparing to known embeddings.
    
    Args:
        frame: Image containing a face
        known_embeddings: Dictionary of person names to embedding lists
        threshold: Maximum distance for a positive match
    
    Returns:
        Tuple of (person_name, embedding) or (None, embedding) if no match
    """
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

# Event logging functions
def save_snapshot(frame, label, x, y, w, h, color):
    """
    Save annotated frame with bounding box and label to disk.
    
    Args:
        frame: Original frame
        label: Text label to display
        x, y, w, h: Bounding box coordinates
        color: BGR color tuple for rectangle and text
    
    Returns:
        Path to saved snapshot file
    """
    annotated = frame.copy()
    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
    cv2.putText(annotated, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    filename = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(SNAPSHOT_DIR, filename)
    cv2.imwrite(path, annotated)
    return path

def broadcast_event(event):
    """
    Add event to queue for WebSocket broadcasting to all connected clients.
    
    Args:
        event: Event dictionary to broadcast
    """
    event_queue.put(event)

def log_event_no_snapshot(camera_id, label):
    """
    Log event without creating a snapshot (used for authorized users).
    
    Args:
        camera_id: Index of camera that detected the event
        label: Description of the event
    
    Returns:
        Event dictionary
    """
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
    """
    Log intruder event with snapshot (includes cooldown to prevent spam).
    
    Args:
        camera_id: Index of camera that detected the event
        label: Description of the event
        frame: Frame to save as snapshot
        x, y, w, h: Bounding box coordinates
        color: BGR color tuple for annotations
    
    Returns:
        Event dictionary
    """
    current_time = time.time()
    should_snapshot = True
    
    # Enforce cooldown period between snapshots for same intruder
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

# WebSocket broadcaster thread
def websocket_broadcaster():
    """
    Background thread that pulls events from queue and broadcasts to all WebSocket clients.
    Automatically removes disconnected clients from the list.
    """
    print("WEBSOCKET: Broadcaster started")
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

# Camera capture thread
def camera_capture_thread(camera_id):
    """
    Dedicated thread for continuously capturing frames from camera.
    Stores frames in shared buffer (camera_frames) for other threads to access.
    This separation prevents resource conflicts when multiple components need camera access.
    
    Args:
        camera_id: Index of camera to capture from
    """
    camera_info = cameras.get(camera_id, {})
    backend = camera_info.get("backend", cv2.CAP_ANY)
    backend_name = camera_info.get("backend_name", "Default")
    
    print(f"CAPTURE: Starting capture thread for camera {camera_id} with {backend_name}")
    
    cap = cv2.VideoCapture(camera_id, backend)
    if not cap.isOpened():
        print(f"Failed to open camera {camera_id}")
        return
    
    # Configure camera settings for optimal performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce latency
    
    print(f"Camera {camera_id} capture started")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read from camera {camera_id}")
            time.sleep(0.1)
            continue
        
        # Update shared frame buffer with thread-safe lock
        with camera_locks[camera_id]:
            camera_frames[camera_id]["frame"] = frame.copy()
            camera_frames[camera_id]["timestamp"] = time.time()
    
    cap.release()

# Face recognition processing thread
def run_camera(camera_id, known_embeddings):
    """
    Process frames from shared buffer for face detection and recognition.
    Reads from camera_frames instead of opening camera directly to prevent conflicts.
    
    Args:
        camera_id: Index of camera to process
        known_embeddings: Dictionary of authorized person embeddings
    """
    global intruder_count
    
    print(f"RECOGNITION: Starting face recognition for camera {camera_id}")
    
    # Load OpenCV face detection cascade classifier
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    face_trackers = {}  # Dictionary tracking detected faces across frames
    TRACKER_TIMEOUT = 2.0  # Seconds before removing inactive face tracker
    frame_count = 0
    PROCESS_EVERY_N_FRAMES = 3  # Process every 3rd frame to reduce CPU usage

    # Wait for first frame from capture thread
    while camera_frames[camera_id]["frame"] is None:
        print(f"Waiting for first frame from camera {camera_id}...")
        time.sleep(0.5)
    
    print(f"Face recognition started for camera {camera_id}")

    while True:
        # Get frame from shared buffer with thread-safe lock
        with camera_locks[camera_id]:
            frame = camera_frames[camera_id]["frame"]
            if frame is None:
                time.sleep(0.1)
                continue
            frame = frame.copy()  # Copy to avoid threading issues

        frame_count += 1
        
        # Skip frames to reduce processing load (process every 3rd frame)
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            time.sleep(0.033)  # Approximately 30 FPS
            continue

        current_time = time.time()
        
        # Remove trackers for faces that haven't been seen recently
        face_trackers = {
            k: v for k, v in face_trackers.items() 
            if current_time - v["last_seen"] < TRACKER_TIMEOUT
        }

        # Detect faces in current frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            # Ignore very small faces (likely false positives)
            if w < 80 or h < 80:
                continue
            
            # Create spatial key for tracking face across frames
            face_key = f"{x//50}_{y//50}"
            
            # Initialize new tracker if this is a new face
            if face_key not in face_trackers:
                face_trackers[face_key] = {
                    "buffer": [],  # Buffer of embeddings for verification
                    "last_seen": current_time,
                    "verified": False,  # Whether identity has been confirmed
                    "label": None  # Assigned label (Authorized/Intruder)
                }
            
            # Calculate servo movement to center face in frame
            frame_cx, frame_cy = frame.shape[1] // 2, frame.shape[0] // 2
            face_cx, face_cy = x + w // 2, y + h // 2
            error_x, error_y = face_cx - frame_cx, face_cy - frame_cy
            send_servo_command(error_x, error_y)
            
            tracker = face_trackers[face_key]
            tracker["last_seen"] = current_time
            
            # Skip recognition if face is already verified
            if tracker["verified"] and tracker["label"]:
                continue
            
            # Extract face region and attempt recognition
            face_crop = frame[y:y+h, x:x+w]
            recognized, emb = recognize_face(face_crop, known_embeddings)

            if recognized:
                # Authorized person detected
                last_seen[recognized] = current_time
                if flags.get(recognized, 0) == 0:
                    flags[recognized] = 1
                    log_event_no_snapshot(camera_id, f"Authorized: {recognized}")
                    print(f"SUCCESS: Authorized: {recognized}")
                
                tracker["verified"] = True
                tracker["label"] = f"Authorized: {recognized}"
                tracker["buffer"] = []

            else:
                # Unknown face detected
                if emb is None:
                    continue
                
                # Check if this matches a known intruder
                matched_intruder = None
                for intruder_id, reps in intruder_embeddings.items():
                    for ref_emb in reps:
                        if cosine_distance(emb, ref_emb) < 0.5:
                            matched_intruder = intruder_id
                            break
                    if matched_intruder:
                        break

                if matched_intruder:
                    # Previously detected intruder
                    if not tracker["verified"]:
                        label = f"Intruder ({matched_intruder})"
                        log_event(camera_id, label, frame, x, y, w, h, (0, 0, 255))
                        print(f"WARNING: Known intruder: {matched_intruder}")
                        tracker["verified"] = True
                        tracker["label"] = label
                        tracker["buffer"] = []
                else:
                    # New unknown face - add to buffer for verification
                    tracker["buffer"].append(emb)
                    
                    # Confirm as new intruder after 8 consistent detections
                    if len(tracker["buffer"]) >= 8:
                        intruder_id = f"intruder_{intruder_count}"
                        intruder_embeddings[intruder_id] = [emb]
                        flags[intruder_id] = -1
                        
                        # Upload intruder data to cloud in background thread
                        threading.Thread(
                            target=save_intruder,
                            args=(intruder_id, np.array(emb), face_crop, cameras[camera_id]["id"]),
                            daemon=True
                        ).start()
                        
                        intruder_count += 1
                        label = f"Intruder ({intruder_id})"
                        log_event(camera_id, label, frame, x, y, w, h, (0, 0, 255))
                        
                        # Send Telegram alert
                        photo_url = supabase.storage.from_("intruder-photos").get_public_url(f"{intruder_id}.jpg")
                        send_telegram_alert(intruder_id, photo_url)

                        print(f"ALERT: NEW INTRUDER: {intruder_id}")
                        
                        tracker["verified"] = True
                        tracker["label"] = label
                        tracker["buffer"] = []
                    else:
                        # Still verifying - broadcast verification status
                        verification_data = {
                            "type": "verification",
                            "camera_id": cameras[camera_id]["id"],
                            "location": cameras[camera_id]["location"],
                            "status": "verifying",
                            "buffer_count": len(tracker["buffer"])
                        }
                        broadcast_event(verification_data)

# FastAPI application setup
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
    Generator function to stream frames from shared buffer as MJPEG.
    Prevents conflicts with capture thread by reading from buffer instead of camera.
    
    Args:
        camera_id: Index of camera to stream from
    
    Yields:
        MJPEG frame bytes in multipart format
    """
    print(f"STREAM: Starting stream for camera {camera_id}")
    
    # Wait for camera to start capturing (with timeout)
    timeout = 10
    start_time = time.time()
    while camera_frames.get(camera_id, {}).get("frame") is None:
        if time.time() - start_time > timeout:
            print(f"ERROR: Timeout waiting for camera {camera_id}")
            
            # Generate placeholder image if camera unavailable
            import io
            from PIL import Image, ImageDraw
            
            img = Image.new('RGB', (640, 480), color=(50, 50, 50))
            draw = ImageDraw.Draw(img)
            draw.text((200, 220), "Camera Unavailable", fill=(255, 255, 255))
            
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Stream placeholder indefinitely
            while True:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + img_byte_arr + b'\r\n')
                time.sleep(1)
            return
        
        time.sleep(0.1)
    
    print(f"SUCCESS: Streaming camera {camera_id}")
    
    while True:
        # Get frame from shared buffer with thread-safe lock
        with camera_locks[camera_id]:
            frame = camera_frames[camera_id]["frame"]
            if frame is None:
                continue
            frame = frame.copy()
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        # Yield frame in multipart format
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # Approximately 30 FPS

@app.get("/camera_feed/{camera_id}")
def camera_feed(camera_id: int):
    """
    Endpoint to stream camera feed as MJPEG.
    
    Args:
        camera_id: Camera number (1-based index, e.g., 1 for first camera)
    
    Returns:
        StreamingResponse with MJPEG video stream
    """
    if camera_id < 1 or camera_id > 10:
        raise HTTPException(status_code=400, detail="Invalid camera ID")
    
    # Convert to 0-based index
    actual_camera_id = camera_id - 1
    
    # Fallback to first available camera if requested camera not found
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
    """
    Initialize system on FastAPI startup:
    1. Build authorized embeddings database
    2. Start WebSocket broadcaster
    3. Launch capture and recognition threads for each camera
    """
    known_embeddings = build_embeddings()
    
    print(f"STARTUP: Starting OmniCam with {len(cameras)} detected camera(s)")
    
    # Start WebSocket broadcaster thread
    threading.Thread(target=websocket_broadcaster, daemon=True).start()
    
    # Start camera capture and processing threads for each detected camera
    for cam_id, cam_info in cameras.items():
        print(f"STARTUP: Starting camera {cam_id} ({cam_info['id']}) - {cam_info['location']}")
        
        # Thread 1: Continuously capture frames from camera
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
    """
    Get list of all detected cameras.
    
    Returns:
        List of camera configuration dictionaries
    """
    return list(cameras.values())

@app.get("/cameras/refresh")
def refresh_cameras():
    """
    Rescan system for available cameras.
    
    Returns:
        Dictionary with refresh status and updated camera list
    """
    global cameras
    cameras = find_working_cameras()
    return {
        "message": f"Refreshed cameras. Found {len(cameras)} working camera(s).",
        "cameras": list(cameras.values())
    }

@app.get("/events")
def get_events():
    """
    Get most recent 50 events.
    
    Returns:
        List of event dictionaries sorted by timestamp
    """
    return list(events)[-50:]

@app.get("/events/latest")
def latest_event():
    """
    Get the most recent event.
    
    Returns:
        Latest event dictionary, or empty dict if no events
    """
    return list(events)[-1] if events else {}

@app.get("/snapshot/{event_id}")
def get_snapshot(event_id: str):
    """
    Retrieve snapshot image for a specific event.
    
    Args:
        event_id: Unique identifier of the event
    
    Returns:
        FileResponse with image, or error dictionary if not found
    """
    for e in events:
        if e["id"] == event_id and e.get("image_path"):
            return FileResponse(e["image_path"])
    return {"error": "Snapshot not found"}

@app.websocket("/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time event streaming.
    Clients connect here to receive live updates about detections.
    
    Args:
        websocket: WebSocket connection object
    """
    await websocket.accept()
    ws_clients.append(websocket)
    print(f"SUCCESS: WebSocket client connected. Total: {len(ws_clients)}")
    try:
        # Keep connection alive by receiving messages
        while True:
            await websocket.receive_text()
    except Exception as e:
        print(f"WebSocket disconnected: {e}")
    finally:
        if websocket in ws_clients:
            ws_clients.remove(websocket)
        print(f"DISCONNECT: WebSocket disconnected. Total: {len(ws_clients)}")

@app.get("/flags")
def get_flags():
    """
    Get current detection flags for all tracked individuals.
    Flag values: 0 = authorized but not yet seen, 1 = authorized and seen, -1 = intruder
    
    Returns:
        Dictionary mapping person/intruder ID to flag value
    """
    return flags

@app.get("/stats")
def get_stats():
    """
    Get system statistics and current status.
    
    Returns:
        Dictionary containing:
        - total_events: Total number of logged events
        - intruder_count: Number of unique intruders detected
        - authorized_count: Number of authorized persons in system
        - active_cameras: Number of cameras currently active
        - ws_clients: Number of connected WebSocket clients
    """
    return {
        "total_events": len(events),
        "intruder_count": intruder_count,
        "authorized_count": len([f for f in flags if flags[f] >= 0]),
        "active_cameras": len(cameras),
        "ws_clients": len(ws_clients)
    }