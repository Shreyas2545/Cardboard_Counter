from flask import Flask, render_template, request
import numpy as np
import cv2

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return "No file uploaded", 400

    file = request.files["image"]

    if file.filename == "":
        return "No file selected", 400

    # ✅ Read image into memory (NO saving)
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return "Invalid image file", 400

    # ✅ Basic OpenCV processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape

    return render_template(
        "result.html",
        message="Image processed successfully!",
        height=height,
        width=width
    )

if __name__ == "__main__":
    app.run(debug=True)
