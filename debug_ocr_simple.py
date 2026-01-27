import pytesseract
from PIL import Image
import os

# Set tesseract path
pytesseract.pytesseract.tesseract_cmd = r"S:\Tesseract-OCR\tesseract.exe"

image_path = r"c:\Users\suman\Downloads\ZS-RAG\my-app\data\books\001-Lamborghini-Huracan-STJ.jpg"

try:
    print(f"Testing OCR on {image_path}")
    text = pytesseract.image_to_string(Image.open(image_path))
    print("OCR Result length:", len(text))
    print("OCR Success!")
except Exception as e:
    print("OCR Failed!")
    print(e)
