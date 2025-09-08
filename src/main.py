import os
import cv2
import numpy as np
from deepface import DeepFace
from collections import deque

AUTHORIZED_DIR = "data/authorized"
flags = {}
intruder_embeddings = {}
intruder_count = 0
intruder_buffer = []
miss_counter = {}
recent_labels = deque(maxlen=5)

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
            print(f"Loaded {len(reps)} embeddings for {person}")
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

def run_face_recognition(camera_index=0):
    global intruder_count, intruder_buffer, miss_counter
    known_embeddings = build_embeddings()
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(camera_index)

    print("Press q to quit")

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

            if recognized:
                intruder_buffer.clear()
                miss_counter[recognized] = 0

                if flags[recognized] == 0:
                    flags[recognized] = 1
                label, color = f"Authorized: {recognized}", (0, 255, 0)

            else:
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
                    if len(intruder_buffer) >= 5:
                        intruder_id = f"intruder_{intruder_count}"
                        intruder_embeddings[intruder_id] = [emb]
                        flags[intruder_id] = -1
                        intruder_count += 1
                        label, color = f"Intruder ({intruder_id})", (0, 0, 255)
                        intruder_buffer.clear()
                    else:
                        label, color = "Checking...", (0, 255, 255)

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
    run_face_recognition(0)
