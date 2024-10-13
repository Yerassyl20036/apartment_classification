import cv2
import easyocr
import re
import os

# Load the EasyOCR reader
reader = easyocr.Reader(['ru'])  # Adjust languages as necessary

# Initialize variables to keep track of the largest number found in multiple runs
largest_overall_number = None

# Function to process the image
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
        print(f"The highest number found is: {highest_number}")
    else:
        print("No valid numbers found.")

# Get image path from user
image_path = input("Enter the path to the image: ")

# Process the image
process_image(image_path)

print("Image processing completed!")
