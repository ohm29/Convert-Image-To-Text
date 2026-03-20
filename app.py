import cv2
import easyocr
import numpy as np
from flask import Flask, render_template, request

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

reader = easyocr.Reader(["en", "th"], gpu=False)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(7, 7))
    cl = clahe.apply(l_channel)

    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


@app.route("/ocr", methods=["GET", "POST"])
def ocr():
    if request.method != "POST":
        return render_template("img2text.html")

    file = request.files.get("file")
    if file is None or file.filename == "":
        return render_template("img2text.html", result="No file selected.")

    if not allowed_file(file.filename):
        return render_template("img2text.html", result="Unsupported file type.")

    try:
        file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return render_template("img2text.html", result="Could not decode image.")

        enhanced = enhance_contrast(image)
        results = reader.readtext(enhanced)
        text = "\n".join(entry[1] for entry in results)
        return render_template("img2text.html", result=text)
    except Exception:
        return render_template("img2text.html", result="Error processing image.")


if __name__ == "__main__":
    app.run()
