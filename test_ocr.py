import pytesseract
from PIL import Image

# Point to tesseract.exe explicitly
pytesseract.pytesseract.tesseract_cmd = r"S:\Tesseract-OCR\tesseract.exe"

img = Image.open(
    r"C:\Users\suman\Downloads\ZS-RAG\my-app\Screenshot 2026-01-27 171822.png"
)

text = pytesseract.image_to_string(img)
print(text)
