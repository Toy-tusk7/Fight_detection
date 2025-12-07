import os
import cv2
import face_recognition
import numpy as np

KNOWN_DIR = "known_faces"

def load_known_faces():
    known_encodings = []
    known_names = []

    for name in os.listdir(KNOWN_DIR):
        person_dir = os.path.join(KNOWN_DIR, name)

        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            path = os.path.join(person_dir, img_name)
            image = face_recognition.load_image_file(path)
            enc = face_recognition.face_encodings(image)

            if len(enc) > 0:
                known_encodings.append(enc[0])
                known_names.append(name)

    print(f"[FACE] Loaded {len(known_names)} known faces.")
    return known_encodings, known_names


def recognize_faces_in_frame(frame, known_encodings, known_names):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb)
    encs = face_recognition.face_encodings(rgb, locs)

    results = []

    for (top, right, bottom, left), enc in zip(locs, encs):
        matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.45)
        face_distances = face_recognition.face_distance(known_encodings, enc)

        if len(face_distances) > 0:
            best_match = np.argmin(face_distances)
            name = known_names[best_match] if matches[best_match] else "Unknown"
        else:
            name = "Unknown"

        face_img = frame[top:bottom, left:right]
        results.append((name, (top, right, bottom, left), face_img))

    return results
