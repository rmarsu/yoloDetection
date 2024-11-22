from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2

with open('types.txt', 'r') as f:
    classes = f.read().strip().split('\n')

model = YOLO()

file = open("image.jpg")

image = cv2.imread("image.jpg")
image = np.asarray(image)

results = model.predict(image)

for result in results:
    for box in result.boxes.data:  # Adjust based on your output structure
        x1, y1, x2, y2, conf, cls = box  # Get coordinates and details
        print(x1, y1, x2, y2, conf, cls)
        if conf > 0.5:  # Confidence threshold
            label = f"Class: {classes[int(cls)]}, Conf: {conf:.2f}"
            # Draw the rectangle
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            # Add label text
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_ITALIC, 0.5, (255, 0, 0), 2)

# Save or display the modified image
cv2.imwrite("result_image.jpg", image)  # Save the result