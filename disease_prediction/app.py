import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model("G:\Artificial Intelligence\disease_prediction\image_classification_cnn_model .h5")

# Define class labels
class_indices = ['BACTERIAL_SOFT_ROT', 'BANANA_APHIDS', 'BANANA_FRUIT_SCARRING_BEETLE',
          'BLACK_SIGATOKA', 'PANAMA_DISEASE', 'PSEUDOSTEM_WEEVIL', 'YELLOW_SIGATOKA']  # Replace with your actual class labels
img_height, img_width = 224, 224  # Dimensions used in model training

# Upload folder
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Check if an image is uploaded
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            # Save uploaded image
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # Predict class
            predicted_class, confidence = predict_image(filepath)

            return render_template(
                "result.html",
                filepath=filepath,
                predicted_class=predicted_class,
                confidence=confidence,
            )
    return render_template("index.html")

def predict_image(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_label = class_indices[predicted_class_index]
    confidence = prediction[0][predicted_class_index] * 100

    return predicted_class_label, confidence

if __name__ == "__main__":
    app.run(debug=True)
