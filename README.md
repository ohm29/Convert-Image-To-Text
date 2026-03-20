# Image2Text

A web service that extracts text from images using [EasyOCR](https://github.com/JaidedAI/EasyOCR). Supports **Thai** and **English** with an optimized image preprocessing pipeline for improved accuracy.

## Features

- Web UI for uploading images and viewing extracted text
- JSON API for programmatic access
- Image preprocessing: auto-scaling, CLAHE contrast enhancement, bilateral denoising, sharpening
- Beam search decoding for higher accuracy on Thai text
- Supports PNG, JPG, JPEG, BMP, TIFF, WebP (max 16 MB)

## Quick Start

### Local

```bash
pip install -r requirements.txt
flask run
```

The app runs at `http://localhost:5000/ocr`.

### Docker

```bash
docker build -t image2text .
docker run -p 5000:5000 image2text
```

## API

### `POST /api/ocr`

Extract text from an uploaded image.

**Request** (`multipart/form-data`):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | Yes | Image file (PNG, JPG, JPEG, BMP, TIFF, WebP) |

**cURL example:**

```bash
curl -X POST http://localhost:5000/api/ocr \
  -F "file=@image.png"
```

**Python example:**

```python
import requests

resp = requests.post(
    "http://localhost:5000/api/ocr",
    files={"file": open("image.png", "rb")},
)
print(resp.json())
```

**Success response** (`200`):

```json
{
  "text": "Extracted text…"
}
```

**Error response** (`400` / `500`):

```json
{
  "error": "Error message"
}
```

## Project Structure

```
├── app.py              # Flask application & OCR logic
├── templates/
│   └── img2text.html   # Web UI
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

## License

MIT
