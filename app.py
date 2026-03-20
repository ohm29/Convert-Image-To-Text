import cv2
import easyocr
import numpy as np
from flask import Flask, jsonify, render_template, request

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}
SUPPORTED_MODELS = {"easyocr", "tesseract"} if TESSERACT_AVAILABLE else {"easyocr"}

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


def upscale_if_small(image: np.ndarray, min_height: int = 1000) -> np.ndarray:
    """Thai glyphs (tone marks, above/below vowels) need sufficient resolution."""
    h, w = image.shape[:2]
    if h < min_height:
        scale = min_height / h
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return image


def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
    image = upscale_if_small(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    image = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, sharp_kernel)

    binary = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )

    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


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
        text_threshold=0.5,
        low_text=0.3,
        contrast_ths=0.1,
        adjust_contrast=0.7,
    )
    return "\n".join(entry[1] for entry in results)


def ocr_tesseract(image: np.ndarray) -> str:
    if not TESSERACT_AVAILABLE:
        raise RuntimeError("Tesseract is not available in this environment.")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    config = "--oem 3 --psm 6"
    return pytesseract.image_to_string(rgb, lang="tha+eng", config=config).strip()


@app.route("/ocr", methods=["GET", "POST"])
def ocr():
    if request.method != "POST":
        return render_template("img2text.html")

    file = request.files.get("file")
    model = request.form.get("model", "easyocr")

    if file is None or file.filename == "":
        return render_template("img2text.html", result="No file selected.", model=model)

    if not allowed_file(file.filename):
        return render_template("img2text.html", result="Unsupported file type.", model=model)

    if model not in SUPPORTED_MODELS:
        return render_template("img2text.html", result="Unsupported model.", model=model)

    try:
        image = read_image(file)
        if image is None:
            return render_template("img2text.html", result="Could not decode image.", model=model)

        text = ocr_easyocr(image) if model == "easyocr" else ocr_tesseract(image)
        return render_template("img2text.html", result=text, model=model)
    except Exception:
        return render_template("img2text.html", result="Error processing image.", model=model)


@app.route("/api/ocr", methods=["POST"])
def api_ocr():
    """JSON API endpoint for OCR processing."""
    file = request.files.get("file")
    model = request.form.get("model", "easyocr")

    if file is None or file.filename == "":
        return jsonify({"error": "No file provided."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type."}), 400

    if model not in SUPPORTED_MODELS:
        return jsonify({"error": f"Unsupported model. Choose from: {', '.join(SUPPORTED_MODELS)}"}), 400

    try:
        image = read_image(file)
        if image is None:
            return jsonify({"error": "Could not decode image."}), 400

        text = ocr_easyocr(image) if model == "easyocr" else ocr_tesseract(image)
        return jsonify({"model": model, "text": text})
    except Exception:
        return jsonify({"error": "Error processing image."}), 500


if __name__ == "__main__":
    app.run()
