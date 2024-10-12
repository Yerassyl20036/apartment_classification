import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# Load the trained YOLOv8 model
model_path = r'.\yolov8_custom313\weights\best.pt'  # Path to your trained model
model = YOLO(model_path)

# Specify the directory containing your test images
image_dir = r'C:\Users\Yerassyl\yolo test\test_images'  # Path to your test images
save_dir = r'C:\Users\Yerassyl\yolo test\output_images'  # Directory to save results

# Create the output directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Get a list of all image files in the test directory
image_files = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]

# Run inference and visualize results for each image
for image_path in image_files:
    print(f"Processing image: {image_path}")
    
    # Run inference without displaying results
    results = model.predict(source=image_path, show=False)  # set show=False to avoid displaying

    # Save the results with bounding boxes and labels
    for result in results:  # Iterate through the results list
        img = cv2.imread(image_path)
        img_with_boxes = result.plot()  # Plot the result on the image
        output_path = os.path.join(save_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, img_with_boxes)  # Save the image with bounding boxes

    # After running inference
    # img = cv2.imread(image_path)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.axis('off')  # Hide axes
    # plt.show()  # Display the image

# Print summary of results (optional)
print("Inference completed on all images!")
