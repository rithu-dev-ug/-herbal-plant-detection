from flask import Flask, render_template, request
import os
import uuid
from werkzeug.utils import secure_filename
from inference_engine import predict_plant

app = Flask(__name__)
UPLOAD_FOLDER      = "static/uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("static/results", exist_ok=True)
# Auto cleanup old result images
import glob
old_results = glob.glob("static/results/result_*.jpg")
for old in old_results[:-10]:  # keep last 10
    try:
        os.remove(old)
    except:
        pass

# ─────────────────────────────────────────────
# FILE VALIDATION
# ─────────────────────────────────────────────
def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ─────────────────────────────────────────────
# MAIN ROUTE
# ─────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():

    results            = []
    error              = None
    result_image_path  = None

    if request.method == "POST":

        if "image" not in request.files:
            error = "No file uploaded."
            return render_template("index.html", results=results, error=error, image=None)

        file = request.files["image"]

        if file.filename == "":
            error = "No file selected."
            return render_template("index.html", results=results, error=error, image=None)

        if not allowed_file(file.filename):
            error = "Invalid file type. Please upload a JPG or PNG image."
            return render_template("index.html", results=results, error=error, image=None)

        filename        = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        save_path       = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        file.save(save_path)

        # Run inference — now returns results, image_path, error
        predictions, result_image_path, err = predict_plant(save_path)

        # Cleanup uploaded file
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except Exception as e:
                print(f"[WARN] Could not delete upload: {e}")

        if err:
            error = err
        elif not predictions:
            error = "No medicinal plant could be confidently identified in this image."
        else:
            results = predictions

    return render_template(
        "index.html",
        results=results,
        error=error,
        image=result_image_path
    )


if __name__ == "__main__":
    app.run(debug=True)