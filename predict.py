import os
import time
from ultralytics import YOLO
import cv2

VIDEOS_DIR = os.path.join('.', 'videos')
OUTPUT_DIR = os.path.join('.', 'output')

video_path = os.path.join(VIDEOS_DIR, 'ra.mp4')
video_path_out = '{}_out.mp4'.format(video_path)
text_file_path = os.path.join(OUTPUT_DIR, 'bounding_boxes.txt')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'runs', 'detect', 'train2', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

class_name_dict = {0: 'Friendly',
                   1: 'Enemy',
                   2: 'Dead Enemy',
                   3: 'Dead Friendly'}

class_color_dict = {0: (0, 255, 0),  # Green for 'Friendly'
                    1: (0, 0, 255),  # Blue for 'Enemy'
                    2: (0, 255, 255),  # Yellow for 'Dead Enemy'
                    3: (255, 0, 0)}  # Red for 'Dead Friendly'

video_fps = cap.get(cv2.CAP_PROP_FPS)
start_time = time.time()

with open(text_file_path, 'w') as text_file:
    while ret:
        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                class_name = class_name_dict[int(class_id)]
                class_color = class_color_dict[int(class_id)]

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), class_color, 4)
                cv2.putText(frame, class_name.upper(), (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, class_color, 3, cv2.LINE_AA)

                # Calculate the relative time
                current_time = time.time() - start_time
                timestamp = time.strftime("%H:%M:%S", time.gmtime(current_time))

                # Write bounding box information to the text file
                text_file.write(f"Time: {timestamp}, Class: {class_name}, X: {x1}, Y: {y1}\n")

        out.write(frame)
        ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()