# src/server.py
import os
import cv2
import numpy as np
import time
import uuid
import threading
from datetime import datetime
from collections import deque
from deepface import DeepFace
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv
import tempfile
import requests

# ----------------- CONFIG -----------------
AUTHORIZED_DIR = "data/authorized"
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

cameras = {
    0: {"id": "cam1", "location": "Lobby"},
}

# ----------------- Globals -----------------
flags = {}
intruder_embeddings = {}
intruder_count = 0
intruder_buffer = []
events = []
label_buffers = {cid: deque(maxlen=5) for cid in cameras.keys()}
recent_labels = deque(maxlen=5)
last_seen = {}
RESET_TIME = 3600  # 1 hour
ws_clients = []

# ----------------- Supabase + Telegram -----------------
load_dotenv()
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

def send_telegram_alert(intruder_id, photo_url):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    message = f"🚨 Intruder Detected!\n\nID: {intruder_id}\nPhoto: {photo_url}"
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": message}
        )
        print("📲 Telegram alert sent")
    except Exception as e:
        print("❌ Telegram failed:", e)

def save_intruder(intruder_id, emb, face_crop, camera_id="cam1"):
    # Save temp image
    tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp_file.name, face_crop)

    # Unique filename for storage
    file_name = f"{intruder_id}_{uuid.uuid4().hex}.jpg"
    file_path = f"intruder-photos/{file_name}"

    # Upload to Supabase storage (no dict options here)
    with open(tmp_file.name, "rb") as f:
        supabase.storage.from_("intruder-photos").upload(file_path, f)

    # Get public URL
    photo_url = supabase.storage.from_("intruder-photos").get_public_url(file_path)

    # Insert event into table
    data = {
        "intruder_id": intruder_id,
        "camera_id": camera_id,
        "embedding": emb.tolist(),
        "photo_url": photo_url
    }
    supabase.table("intruders").insert(data).execute()

    print(f"☁️ Logged intruder {intruder_id} with photo → {photo_url}")
    send_telegram_alert(intruder_id, photo_url)
    

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
                    enforce_detection=False
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

# ----------------- SNAPSHOT + EVENTS -----------------
def save_snapshot(frame, detections):
    annotated = frame.copy()
    for det in detections:
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]
        label, color = det["label"], det["color"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(annotated, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
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
    cap = cv2.VideoCapture(camera_id)
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
                intruder_buffer.clear()
                last_seen[recognized] = time.time()
                if flags[recognized] == 0:
                    flags[recognized] = 1
                label, color = f"Authorized: {recognized}", (0, 255, 0)
            else:
                # reset inactive authorized
                for person, status in flags.items():
                    if status == 1 and person in last_seen:
                        if time.time() - last_seen[person] > RESET_TIME:
                            flags[person] = 0
                            print(f"⏳ Reset {person} (inactive >{RESET_TIME}s)")
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
                    if len(intruder_buffer) >= 8:
                        intruder_id = f"intruder_{intruder_count}"
                        intruder_embeddings[intruder_id] = [emb]
                        flags[intruder_id] = -1
                        save_intruder(intruder_id, np.array(emb), face_crop)
                        intruder_count += 1
                        label, color = f"Intruder ({intruder_id})", (0, 0, 255)
                        intruder_buffer.clear()
                        print(f"⚠️ Intruder detected: {intruder_id}")
                    else:
                        label, color = "Verifying...", (0, 255, 255)
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

# ----------------- FASTAPI APP -----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
