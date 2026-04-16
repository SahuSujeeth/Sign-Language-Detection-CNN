import cv2
import os

label = input("Enter label (A-Z or 0-9): ")

dataset_path = "dataset/" + label

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

cap = cv2.VideoCapture(0)

count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Flip (same everywhere)
    frame = cv2.flip(frame, 1)

    # ROI box
    cv2.rectangle(frame, (100, 100), (350, 350), (0, 255, 0), 2)

    roi = frame[100:350, 100:350]

    # 🔥 NORMAL IMAGE (NO THRESHOLD)
    cv2.imshow("Frame", frame)
    cv2.imshow("ROI", roi)

    key = cv2.waitKey(1)

    if key == ord('s'):
        img_name = f"{dataset_path}/{count}.jpg"
        cv2.imwrite(img_name, roi)
        count += 1
        print(f"Saved {count}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()