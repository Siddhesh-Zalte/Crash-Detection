import cv2
import numpy as np
import pygame
import time

# Initialize pygame for audio
pygame.mixer.init()
sound = pygame.mixer.Sound("audio1.mp3")

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load video
video = cv2.VideoCapture("road_car_view.mp4")

frame_count = 0
last_play_time = 0  # Initialize the last play time
sound_interval = 0.25  # Interval between sound plays in seconds

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 5 != 0:  # Process every 5th frame
        continue

    # Resize frame
    scale_factor = 0.5
    height, width, _ = frame.shape
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    frame = cv2.resize(frame, (new_width, new_height))

    # Preprocess frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Get vehicle detections
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id in [2, 3, 7]:  # 2 = car, 3 = motorbike, 7 = truck
                center_x = int(detection[0] * new_width)
                center_y = int(detection[1] * new_height)
                w = int(detection[2] * new_width)
                h = int(detection[3] * new_height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression to remove duplicate boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    red_box_shown = False

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            distance_estimate = new_height / h  # Simple distance estimation

            # Determine color and label based on class_id
            if class_ids[i] == 2:
                label = "Car"
            elif class_ids[i] == 3:
                label = "Bike"
            elif class_ids[i] == 7:
                label = "Truck"
            
            if distance_estimate < 2.5:
                color = (0, 0, 255)  # Red box for close vehicle
                red_box_shown = True
            else:
                color = (0, 255, 0)  # Green box for regular detection

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} Distance: {distance_estimate:.2f}m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Play sound if red box is shown and enough time has passed
    if red_box_shown:
        current_time = time.time()
        if current_time - last_play_time > sound_interval:
            sound.play()
            last_play_time = current_time  # Update the last play time

    # Show the frame with detected vehicles
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()
