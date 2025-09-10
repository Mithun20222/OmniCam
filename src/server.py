# src/server.py
import os
import cv2
import numpy as np
from deepface import DeepFace
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from datetime import datetime
import threading, uuid
from collections import deque

# ------------- CONFIG -------------
AUTHORIZED_DIR = "data/authorized"
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Cameras to track
cameras = {
    0: {"id": "cam1", "location": "Lobby"},
}

# Memory
flags = {}
intruder_embeddings = {}
intruder_count = 0
events = []   # list of dicts (for REST + WebSocket)

# Add per-camera stability buffers
label_buffers = {cid: deque(maxlen=5) for cid in cameras.keys()}

# ------------- FACE UTILS -------------
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
                    enforce_detection=False
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
            enforce_detection=False
        )[0]["embedding"]

        best_match, best_score = None, 1e6
        for person, reps in known_embeddings.items():
            for ref_emb in reps:
                dist = cosine_distance(emb, ref_emb)
                if dist < best_score:
                    best_score, best_match = dist, person

        return (best_match, emb) if best_score < threshold else (None, emb)
    except:
        return None, None

# ------------- SNAPSHOT / EVENT LOGGING -------------
def save_snapshot(frame, detections):
    annotated = frame.copy()
    for det in detections:
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]
        label, color = det["label"], det["color"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            annotated, label, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    filename = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(SNAPSHOT_DIR, filename)
    cv2.imwrite(path, annotated)
    return path

def log_event(camera_id, detections, frame):
    stable_label = detections[0]["label"] if detections else "Unknown"
    event = {
        "id": uuid.uuid4().hex,
        "camera_id": cameras[camera_id]["id"],
        "location": cameras[camera_id]["location"],
        "timestamp": datetime.utcnow().isoformat(),
        "label": stable_label,
        "image_path": save_snapshot(frame, detections)
    }
    events.append(event)
    # Send via WebSocket
    for ws in ws_clients:
        try:
            import asyncio
            asyncio.run(ws.send_json(event))
        except:
            pass
    return event

# ------------- CAMERA UTILS -------------
def open_camera(camera_id):
    """Try DirectShow first, fallback to MSMF."""
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"✅ Camera {camera_id} opened with DirectShow")
        return cap
    cap = cv2.VideoCapture(camera_id, cv2.CAP_MSMF)
    if cap.isOpened():
        print(f"⚠️ Camera {camera_id} opened with MSMF")
        return cap
    print(f"❌ Camera {camera_id} could not be opened")
    return None

# ------------- CAMERA LOOP -------------
def run_camera(camera_id, known_embeddings):
    global intruder_count
    cap = open_camera(camera_id)
    if cap is None:
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        detections = []

        for (x, y, w, h) in faces:
            if w < 80 or h < 80:
                continue

            face_crop = frame[y:y+h, x:x+w]
            recognized, emb = recognize_face(face_crop, known_embeddings)

            if recognized:
                label, color = f"Authorized: {recognized}", (0, 255, 0)
                flags[recognized] = 1
            else:
                label, color = "Intruder", (0, 0, 255)
                intruder_id = f"intruder_{intruder_count}"
                if intruder_id not in intruder_embeddings:
                    intruder_embeddings[intruder_id] = [emb]
                    flags[intruder_id] = -1
                    intruder_count += 1

            detections.append({
                "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "label": label, "color": color
            })

        if detections:
            label_buffers[camera_id].append(detections[0]["label"])
            stable_label = max(set(label_buffers[camera_id]),
                               key=label_buffers[camera_id].count)
            detections[0]["label"] = stable_label
            log_event(camera_id, detections, frame)

    cap.release()

# ------------- FASTAPI APP -------------
app = FastAPI()
ws_clients = []

@app.on_event("startup")
def startup_event():
    known_embeddings = build_embeddings()
    for cam_id in cameras.keys():
        threading.Thread(target=run_camera,
                         args=(cam_id, known_embeddings),
                         daemon=True).start()

@app.get("/cameras")
def get_cameras():
    return list(cameras.values())

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
            return FileResponse(e["image_path"])
    return {"error": "Not found"}

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
