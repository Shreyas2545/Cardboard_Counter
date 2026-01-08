from flask import Flask, render_template, request
import numpy as np
import cv2

from cv_engine.preprocessing import preprocess_image
from cv_engine.gradients import compute_directional_gradient
from cv_engine.projection import project_to_1d
from cv_engine.counter import count_cardboards

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

    # In-memory image read
    image_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return "Invalid image", 400

    # CV Pipeline
    preprocessed = preprocess_image(image)
    gradient = compute_directional_gradient(preprocessed, "vertical")
    signal = project_to_1d(gradient, "vertical")

    cardboard_count, flute_count = count_cardboards(signal)

    return render_template(
        "result.html",
        count=cardboard_count,
        flutes=flute_count
    )

if __name__ == "__main__":
    app.run(debug=True)
