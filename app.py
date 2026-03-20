import cv2
import easyocr
import numpy as np
from flask import Flask, jsonify, render_template, request

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

_reader = None


def get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["th", "en"], gpu=False)
    return _reader


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


MIN_HEIGHT = 1000
MAX_DIMENSION = 4000


def _rescale(image: np.ndarray) -> np.ndarray:
    """Thai glyphs (tone marks, above/below vowels) need sufficient resolution,
    but excessively large images waste time without improving accuracy."""
    h, w = image.shape[:2]
    if h < MIN_HEIGHT:
        scale = MIN_HEIGHT / h
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        h, w = image.shape[:2]
    if max(h, w) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
    image = _rescale(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    sharp_kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    sharpened = cv2.filter2D(gray, -1, sharp_kernel)

    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def read_image(file) -> np.ndarray | None:
    file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return enhance_for_ocr(image)


def ocr_easyocr(image: np.ndarray) -> str:
    results = get_reader().readtext(
        image,
        paragraph=True,
        decoder="beamsearch",
        beamWidth=10,
        text_threshold=0.5,
        low_text=0.3,
        contrast_ths=0.1,
        adjust_contrast=0.7,
    )
    return "\n".join(entry[1] for entry in results)


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
        image = read_image(file)
        if image is None:
            return render_template("img2text.html", result="Could not decode image.")

        text = ocr_easyocr(image)
        return render_template("img2text.html", result=text)
    except Exception:
        return render_template("img2text.html", result="Error processing image.")


@app.route("/api/ocr", methods=["POST"])
def api_ocr():
    """JSON API endpoint for OCR processing."""
    file = request.files.get("file")

    if file is None or file.filename == "":
        return jsonify({"error": "No file provided."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type."}), 400

    try:
        image = read_image(file)
        if image is None:
            return jsonify({"error": "Could not decode image."}), 400

        text = ocr_easyocr(image)
        return jsonify({"text": text})
    except Exception:
        return jsonify({"error": "Error processing image."}), 500


if __name__ == "__main__":
    app.run()
