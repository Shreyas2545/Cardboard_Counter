from flask import Flask, render_template, request

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

    # For now, just confirm receipt
    return render_template(
        "result.html",
        message="Image received successfully!"
    )

if __name__ == "__main__":
    app.run(debug=True)

