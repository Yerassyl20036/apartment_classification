from ultralytics import YOLO

# Load a pretrained YOLOv8 model (e.g., yolov8n.pt)
model = YOLO('yolov8n.pt')  # YOLOv8n, adjust if you need a different model

# Train the model
if __name__ == "__main__":
    model.train(
        data=r'C:\Users\Yerassyl\yolo test\krisha.kz.v5i.yolov8\data.yaml', 
        imgsz=640,  # Image size for training
        epochs=50,  # Number of epochs
        batch=16,  # Batch size
        name='yolov8_custom3',  # Name of training run
        device=0  # Set to GPU 0, or 'cpu' for training on CPU
    )
