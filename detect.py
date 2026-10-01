import cv2
from ultralytics import YOLO

model_path = 'runs/detect/train2/weights/best.pt' 
model = YOLO(model_path)

def detect_plate(image_path):
    img = cv2.imread(image_path)
    results = model(img)
    plates = []
    for r in results:
        boxes = r.boxes.xyxy
        for box in boxes:
            x1,y1,x2,y2 = map(int, box)
            plate = img[y1:y2, x1:x2]
            plates.append(plate)
    return plates