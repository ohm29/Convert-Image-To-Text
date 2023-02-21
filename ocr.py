import cv2 
import pytesseract
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
tessdata_dir_config = '-c preserve_interword_spaces=1  --dpi 300'



# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def ocr(image):
    image = cv2.imread(image, 1)
    ray = get_grayscale(image)
    thresh = thresholding(gray)

    plt.imshow(thresh)
    text = pytesseract.pytesseract.image_to_string(thresh,lang='eng+tha', config=tessdata_dir_config)
    print(text)