from ultralytics import YOLO

# Step 1: Load the YOLOv8 model (you can specify a pre-trained model like yolov8n.pt)
model = YOLO('yolov8n.pt')  # or specify a custom model path

# Step 2: Specify the paths to the krisha.kz.v3i.yolov8 and configure the training
data_config = {
    'train': './krisha.kz.v3i.yolov8/train',  # Path to your training images
    'val': './krisha.kz.v3i.yolov8/valid',      # Path to your validation images
    'test': './krisha.kz.v3i.yolov8/test',    # Path to your test images (optional, for later evaluation)
    'nc': 7,                    # Number of classes in your krisha.kz.v3i.yolov8
    'names': ['balcony', 'bathroom', 'hallway', 'kitchen', 'living_room', 'other', 'studio'],  # List of class names
}

# Step 3: Train the model using your krisha.kz.v3i.yolov8
results = model.train(
    data=data_config,   # Pass the data configuration
    epochs=41,         # Number of training epochs
    imgsz=640           # Image size for training
)

# Step 4: Evaluate the model on the test set after training
metrics = model.val(
    data='./krisha.kz.v3i.yolov8/test'  # Path to your test dataset
)

# Step 5: Print out evaluation metrics (mAP, Precision, Recall)
print(metrics)
