import face_recognition
import cv2
import numpy as np
import os

# --- 1. Load Known Faces ---
print("Loading known faces...")
known_face_encodings = []
known_face_names = []

known_faces_dir = "known_faces"

for filename in os.listdir(known_faces_dir):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(known_faces_dir, filename)
        known_image = face_recognition.load_image_file(image_path)

        try:
            encoding = face_recognition.face_encodings(known_image)[0]
            name = os.path.splitext(filename)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(name)
        except IndexError:
            print(f"Warning: No face found in {filename}. Skipping.")

print(f"Loaded {len(known_face_names)} known faces.")

# --- 2. Initialize Webcam ---
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # --- 3. Grab a Single Frame of Video ---
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # --- 4. Process the Frame ---
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all the faces, encodings, AND landmarks in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # *** NEW: Get face landmarks ***
    face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        face_names.append(name)

    # --- 5. Display the Results ---
    # We zip all three lists to loop through them together
    for (top, right, bottom, left), name, landmarks in zip(face_locations, face_names, face_landmarks_list):
        # Scale back up face locations
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # *** NEW: Draw Face Landmarks ***
        # Loop over each facial feature (eye, nose, mouth, etc)
        for feature_name, points in landmarks.items():
            # Convert the list of (x, y) tuples to a NumPy array
            # and scale it back up to the original frame size
            points_np = (np.array(points) * 4).astype(int)

            # Draw the lines for the feature
            # isClosed=False for features like 'nose_bridge'
            # isClosed=True for features like 'left_eye'
            is_closed = True if "eye" in feature_name or "lip" in feature_name else False
            cv2.polylines(frame, [points_np], isClosed=is_closed, color=(255, 0, 0), thickness=1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 6. Release Handle to the Webcam ---
video_capture.release()
cv2.destroyAllWindows()