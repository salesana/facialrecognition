import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = "./face_dataset/"

# Ensure dataset_path directory exists
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
    print(f"Created directory: {dataset_path}")

file_name = input("Enter the name of person: ").strip()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(faces) == 0:
        continue

    k = 1

    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

    skip += 1

    for face in faces[:1]:
        x, y, w, h = face

        offset = 5
        face_offset = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_selection = cv2.resize(face_offset, (100, 100))

        if skip % 10 == 0:
            face_data.append(face_selection)
            print(len(face_data))

        cv2.imshow(str(k), face_selection)
        k += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("faces", frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

# Save face data
face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

np.save(dataset_path + file_name, face_data)
print(f"Dataset saved at: {dataset_path + file_name}.npy")

cap.release()
cv2.destroyAllWindows()
