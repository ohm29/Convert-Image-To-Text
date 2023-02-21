from flask import Flask,render_template,request
import cv2 
import pytesseract
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
tessdata_dir_config = '-c preserve_interword_spaces=1  --dpi 300'
import os

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def main(file):
    local_path = os.path.dirname(os.path.realpath(__file__))
    file.save(local_path+'/img/'+str(file.filename))
    img = cv2.imread(local_path+'/img/test.png', 1)
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(7,7))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl,a,b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    gray = get_grayscale(enhanced_img)
    thresh = thresholding(gray)
    txt = pytesseract.pytesseract.image_to_string(thresh,lang='eng+tha', config=tessdata_dir_config)
    print(txt)

