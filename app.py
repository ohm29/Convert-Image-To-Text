
from flask import Flask,render_template,request
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


app = Flask(__name__, template_folder='templates')

@app.route('/ocr', methods=['GET', 'POST'])
def ocr():
    if request.method == 'POST':
        file = request.files['file']
        file.save('image/'+str(file.filename))
        image = cv2.imread('image/'+str(file.filename), 1)
        lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(7,7))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl,a,b))
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        gray = get_grayscale(enhanced_img)
        thresh = thresholding(gray)
        text = pytesseract.pytesseract.image_to_string(thresh,lang='eng+tha', config=tessdata_dir_config)
        return render_template('img2text.html',result=text)
    else:
        return render_template('img2text.html')
 
    
    
# main driver function
if __name__ == '__main__':

    app.run()