import os
import cv2
import numpy as np
import time
from deepface import DeepFace
from collections import deque
from supabase import create_client, Client
from dotenv import load_dotenv
import tempfile
import uuid
import requests

def send_telegram_alert(intruder_id, photo_url):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    message = f"🚨 Intruder Detected!\n\nID: {intruder_id}\nPhoto: {photo_url}"
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": message}
        )
        print("📲 Telegram alert sent")
    except Exception as e:
        print("❌ Telegram failed:", e)

load_dotenv()
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

if not url or not key:
    raise ValueError("Supabase credentials not found. Check your .env file.")

supabase: Client = create_client(url, key)

def save_intruder(intruder_id, emb, face_crop, camera_id="cam_0"):
    # Save temp image
    tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp_file.name, face_crop)

    # Unique filename for storage
    file_name = f"{intruder_id}_{uuid.uuid4().hex}.jpg"
    file_path = f"intruder-photos/{file_name}"

    # Upload to Supabase storage
    with open(tmp_file.name, "rb") as f:
        supabase.storage.from_("intruder-photos").upload(file_path, f, {"upsert": True})

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


# ---------------- Globals ----------------
AUTHORIZED_DIR = "data/authorized"
flags = {}                  
intruder_embeddings = {}    
intruder_count = 0          
intruder_buffer = []        
recent_labels = deque(maxlen=5)
last_seen = {}              
RESET_TIME = 3600  # 1 hour

# ---------------- Step 1: Build embeddings ----------------
def build_embeddings():
    embeddings = {}
    for person in os.listdir(AUTHORIZED_DIR):
        person_dir = os.path.join(AUTHORIZED_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        reps = []
        for img in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img)
            try:
                rep = DeepFace.represent(
                    img_path=img_path,
                    model_name="SFace",
                    detector_backend="opencv",
                    enforce_detection=False
                )[0]["embedding"]
                reps.append(rep)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        if reps:
            embeddings[person] = reps
            flags[person] = 0
            print(f"✅ Loaded {len(reps)} embeddings for {person}")
    return embeddings

# ---------------- Step 2: Cosine similarity ----------------
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

        best_match = None
        best_score = 1e6
        for person, reps in known_embeddings.items():
            for ref_emb in reps:
                dist = cosine_distance(emb, ref_emb)
                if dist < best_score:
                    best_score = dist
                    best_match = person

        if best_score < threshold:
            return best_match, emb
        else:
            return None, emb
    except Exception as e:
        print("Recognition error:", e)
    return None, None

# ---------------- Step 3: Run recognition ----------------
def run_face_recognition(camera_index=0):
    global intruder_count, intruder_buffer
    known_embeddings = build_embeddings()
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(camera_index)

    print("✅ Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        label, color = "No Face", (200, 200, 200)

        for (x, y, w, h) in faces:
            if w < 80 or h < 80:
                continue

            face_crop = frame[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue

            recognized, emb = recognize_face(face_crop, known_embeddings)

            if recognized:  # ✅ Authorized face
                intruder_buffer.clear()
                last_seen[recognized] = time.time()

                if flags[recognized] == 0:
                    flags[recognized] = 1
                label, color = f"Authorized: {recognized}", (0, 255, 0)

            else:
                # ✅ Respect flags: if already authorized, skip intruder logic
                authorized_found = False
                for person, status in flags.items():
                    if status == 1:
                        label, color = f"Authorized: {person}", (0, 255, 0)
                        authorized_found = True
                        break

                if not authorized_found:
                    # Reset flags if last seen > 1 hr
                    for person, status in flags.items():
                        if status == 1 and person in last_seen:
                            if time.time() - last_seen[person] > RESET_TIME:
                                flags[person] = 0
                                print(f"⏳ Resetting {person}'s flag to 0 (inactive >1 hr)")

                    # Intruder logic
                    matched_intruder = None
                    for intruder_id, reps in intruder_embeddings.items():
                        for ref_emb in reps:
                            dist = cosine_distance(emb, ref_emb)
                            if dist < 0.5:
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

            # Smooth labels
            recent_labels.append(label)
            stable_label = max(set(recent_labels), key=recent_labels.count)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(
                frame, stable_label, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        cv2.imshow("Face Recognition (Stable)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Final Flags:", flags)

if __name__ == "__main__":
    run_face_recognition(0)  # change to your webcam index
