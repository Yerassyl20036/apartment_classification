from flask import Flask, request, render_template, redirect, url_for
from ultralytics import YOLO
import re
import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import os
import base64
from io import BytesIO

# Load the trained YOLOv8 model
model_path = r'.\yolov8_custom314\weights\best.pt'  # Path to your trained model
model = YOLO(model_path)
reader = easyocr.Reader(['ru'])

app = Flask(__name__)

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
    
    # Convert the image to base64 for web display
    _, buffer = cv2.imencode('.png', img_with_boxes)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return living_room_count, img_base64

def process_image(image_path):
    global largest_overall_number  # Use the global variable to track across runs
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Image not found at {image_path}")
        return None

    results = reader.readtext(image)
    detected_texts = []
    largest_number = None
    largest_box = None

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
            detected_texts.append(text)

            if number is not None:
                if largest_number is None or number > largest_number:
                    largest_number = number
                    largest_box = bbox

    highest_number = compare(detected_texts)
    if highest_number is not None:
        print(f"The apartment area is: {highest_number}")
    else:
        print("No valid numbers found.")

    return highest_number

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
        return max(valid_numbers)
    else:
        return None

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image_path = os.path.join('uploads', file.filename)
            file.save(image_path)

            living_room_count, img_base64 = process_image1(image_path)
            largest_number = process_image(image_path)
            result = {
                "living_room_count": living_room_count,
                "largest_number": largest_number,
                "processed_image": img_base64
            }
            print(largest_number)
            return render_template('result.html', result=result)

    return render_template('upload.html')

if __name__ == '__main__':
    # Create uploads folder if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host='0.0.0.0', port=5000)
