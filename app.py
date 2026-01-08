from flask import Flask, render_template, request
import numpy as np
import cv2
import base64

from cv_engine.preprocessing import preprocess_image
from cv_engine.gradients import compute_directional_gradient
from cv_engine.projection import project_to_1d
from cv_engine.counter import estimate_sheet_count_with_confidence

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return "No image uploaded", 400

    file = request.files["image"]
    if file.filename == "":
        return "No image selected", 400

    # Read image in memory
    image_bytes = file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if image is None:
        return "Invalid image", 400

    # ----- CV PIPELINE -----
    preprocessed = preprocess_image(image)
    gradient = compute_directional_gradient(preprocessed, "vertical")
    signal_1d = project_to_1d(gradient, "vertical")

    count, confidence = estimate_sheet_count_with_confidence(signal_1d)

    # ----- Convert image to Base64 for display -----
    _, buffer = cv2.imencode(".jpg", image)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return render_template(
        "result.html",
        count=count,
        confidence=confidence,
        image_data=img_base64
    )


if __name__ == "__main__":
    app.run(debug=True)
