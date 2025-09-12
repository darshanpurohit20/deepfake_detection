import os
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

MODEL_PATH = 'deepfake_detector_model.h5'
model = None

def load_app_model():
    """Load the trained model from disk into memory."""
    global model
    try:
        model = load_model(MODEL_PATH)
        print(f"✅ Model '{MODEL_PATH}' loaded successfully")
    except Exception as e:
        print(f"❌ ERROR: Could not load model. Check path.\n{e}")
        exit(1)

def prepare_image(image, target_size):
    """Preprocesses the uploaded image for the model."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0
    return image_array

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded yet.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading.'}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        # 🔧 Make sure this matches your training size
        processed_image = prepare_image(image, target_size=(96, 96))
        prediction = model.predict(processed_image)
        
        confidence_score = float(prediction[0][0])
        is_fake = confidence_score < 0.5
        result_label = "Fake" if is_fake else "Real"
        confidence_percent = (1 - confidence_score if is_fake else confidence_score) * 100

        return jsonify({
            'prediction': result_label,
            'confidence': f'{confidence_percent:.2f}%'
        })

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

# --- IMPORTANT: Load model immediately so it's ready before first request ---
load_app_model()

if __name__ == '__main__':
    # Disable Flask reloader for Colab / ngrok
    app.run(debug=False, use_reloader=False, host="0.0.0.0", port=9129)
