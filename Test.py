import cv2
import glob
import matplotlib.pyplot as plt
import random
from ultralytics import YOLO

model_path = 'runs/detect/train2/weights/best.pt' 
model = YOLO(model_path)
test_images = glob.glob('D:\\Users\\Name\\.cache\\kagglehub\\datasets\\sujaymann\\car-number-plate-dataset-yolo-format\\versions\\3\\License-Plate-Data\\test\\images\\*')
# image_path = "D:\\Test2.jpg"
image_path = random.choice(test_images)
results = model(image_path, conf=0.25)
result_img = results[0].plot() 
img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.imshow(img_rgb)
plt.axis('off')
plt.title(f"Detected")
plt.show()