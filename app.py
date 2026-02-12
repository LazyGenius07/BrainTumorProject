import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# ---------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ---------------------------------------------------------------
# Load Models (.h5)
# ---------------------------------------------------------------
CLASSIFICATION_MODEL_PATH = r"D:\Mini Project\Models\classification_model.h5"
SEGMENTATION_MODEL_PATH   = r"D:\Mini Project\Models\segmentation_model.h5"

# Load models without compiling to avoid H5 issues
classifier = load_model(CLASSIFICATION_MODEL_PATH, compile=False)
segmenter  = load_model(SEGMENTATION_MODEL_PATH, compile=False)

print(" Models loaded successfully!")

# ---------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------
def classify_tumor(image_path):
    """Predict tumor type"""
    img = load_img(image_path, target_size=(128, 128))  # Match model input
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = classifier.predict(img)
    class_idx = np.argmax(preds)

    labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    return labels[class_idx], float(np.max(preds))


def get_tumor_percentage(image_path):
    """Predict tumor mask, save mask & overlay, calculate percentage"""
    img = load_img(image_path, target_size=(128, 128))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    mask = segmenter.predict(img)[0, :, :, 0]
    mask = (mask > 0.5).astype(np.uint8)

    tumor_pixels = np.sum(mask)
    total_pixels = mask.size
    percentage = (tumor_pixels / total_pixels) * 100

    mask_path, overlay_path = save_mask_and_overlay(image_path, mask)

    return round(percentage, 2), mask_path, overlay_path


def save_mask_and_overlay(image_path, mask):
    """Save binary mask and overlay image"""

    filename = os.path.basename(image_path)

    # ----- Save binary mask -----
    mask_img = (mask * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_img)

    mask_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        "mask_" + filename
    )
    mask_pil.save(mask_path)

    # ----- Create overlay -----
    original = load_img(image_path, target_size=(128, 128))
    original = np.array(original)

    overlay = np.zeros_like(original)
    overlay[:, :, 0] = mask * 255  # Red tumor region

    blended = (0.7 * original + 0.3 * overlay).astype(np.uint8)

    overlay_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        "overlay_" + filename
    )
    Image.fromarray(blended).save(overlay_path)

    return mask_path, overlay_path


treatment_info = {
    'Glioma': {
        'treatment': 'Surgery + Radiation + Chemotherapy',
        'prognosis': '5-year survival ~50–80% (depends on grade)'
    },
    'Meningioma': {
        'treatment': 'Surgical removal ± radiation',
        'prognosis': 'Usually benign; >85% 10-year survival'
    },
    'Pituitary': {
        'treatment': 'Surgery or hormone therapy',
        'prognosis': 'High cure rate; often benign'
    },
    'No Tumor': {
        'treatment': 'No treatment required',
        'prognosis': 'Healthy brain (no visible tumor)'
    }
}


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")



@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return redirect(url_for("index"))

    img_file = request.files["image"]
    if img_file.filename == "":
        return redirect(url_for("index"))

    # Save uploaded file
    path = os.path.join(app.config["UPLOAD_FOLDER"], img_file.filename)
    img_file.save(path)

    # Run predictions
    tumor_type, confidence = classify_tumor(path)
    tumor_percent, mask_path, overlay_path = get_tumor_percentage(path)


    # Fetch treatment info
    info = treatment_info.get(tumor_type, {'treatment':'Unknown', 'prognosis':'Unknown'})

    return render_template(
        "result.html",
        image_path=path,
        mask_path=mask_path,
        overlay_path=overlay_path,
        tumor_type=tumor_type,
        confidence=round(confidence * 100, 2),
        tumor_percent=tumor_percent,
        treatment=info['treatment'],
        prognosis=info['prognosis']
    )



# ---------------------------------------------------------------
# Run App
# ---------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
