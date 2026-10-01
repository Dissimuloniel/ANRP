from ultralytics import YOLO

def train_model():
    model = YOLO('yolov8n.pt')
    results = model.train(
        data="data_yolo_friendly.yaml",
        epochs=30,          
        imgsz=640,
        batch=16,
        plots=True
    )
    return results

if __name__ == '__main__':
    train_model()