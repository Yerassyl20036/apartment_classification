from ultralytics import YOLO
import re
import cv2
import easyocr
import matplotlib.pyplot as plt

# Load the trained YOLOv8 model
model_path = r'.\yolov8_custom314\weights\best.pt'  # Path to your trained model
model = YOLO(model_path)

reader = easyocr.Reader(['ru'])
largest_overall_number = None

def process_image1(image_path):
    print(f"Processing image: {image_path}")
    
    # Run inference
    results = model.predict(source=image_path, show=False)

    # Count living_room objects
    living_room_count = sum(1 for box in results[0].boxes if results[0].names[int(box.cls)] == 'living_room')
    print(f"Number of living_room objects detected: {living_room_count}")

    # Load the image
    img = cv2.imread(image_path)

    # Plot the YOLO results on the image
    img_with_boxes = results[0].plot()

    # Display the image with bounding boxes using matplotlib
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Processed Image')
    plt.show()

def process_image(image_path):
    global largest_overall_number  # Use the global variable to track across runs
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Image not found at {image_path}")
        return

    results = reader.readtext(image)

    largest_number = None
    largest_box = None
    detected_texts = []  # Array to store detected texts

    # Regular expression to match text in the format: 2-3 digits followed by a dot/comma, then 1-2 digits
    number_pattern = re.compile(r'\b\d{2,3}[.,]\d{1,2}\b')

    def extract_number(text):
        text = text.replace(',', '.')
        try:
            return float(text)
        except ValueError:
            return None

    for (bbox, text, confidence) in results:
        if number_pattern.search(text):
            number = extract_number(text)
            detected_texts.append(text)  # Add detected text to the array

            if number is not None:
                if largest_number is None or number > largest_number:
                    largest_number = number
                    largest_box = bbox

    compare(detected_texts)  # Pass the detected texts to the compare method

def compare(texts):
    def custom_float_parser(text):
        result = ""
        decimal_encountered = False
        for char in text:
            if char.isdigit():
                result += char
            elif (char == '.' or char == ',') and not decimal_encountered:
                result += '.'
                decimal_encountered = True
            else:
                break
        return float(result) if result else None

    parsed_numbers = [custom_float_parser(text) for text in texts]
    valid_numbers = [num for num in parsed_numbers if num is not None]

    if valid_numbers:
        highest_number = max(valid_numbers)
        print(f"The aparment area is: {highest_number}")
    else:
        print("No valid numbers found.")

# Get image path from user
image_path = input("Enter the path to the image: ")

# Process the image
process_image1(image_path)
process_image(image_path)

print("Image processing completed!")
