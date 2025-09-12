import os
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io

# Initialize the Flask application
app = Flask(__name__)

# Define the path to the model
# It assumes the model is in the same directory as this script
MODEL_PATH = 'deepfake_detector_model.h5' 
model = None

def load_app_model():
    """Load the trained model from disk into memory."""
    global model
    try:
        model = load_model(MODEL_PATH)
        print(f" * Model {MODEL_PATH} loaded successfully")
    except Exception as e:
        print(f" * ERROR: Could not load model. Please check the path. \n{e}")
        exit() # The app can't run without the model

def prepare_image(image, target_size):
    """Preprocesses the uploaded image for the model."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize to the model's expected input size
    image = image.resize(target_size)
    
    # Convert image to a numpy array
    image_array = img_to_array(image)
    
    # Expand dimensions to create a batch of 1
    image_array = np.expand_dims(image_array, axis=0)
    
    # Normalize the image data (rescale pixel values from 0-255 to 0-1)
    image_array /= 255.0
    
    return image_array

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """Render the main HTML page from the 'templates' folder."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to receive an image and return a prediction."""
    if model is None:
        return jsonify({'error': 'Model is not loaded yet.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading.'}), 400

    if file:
        try:
            # Read image directly from the upload
            image = Image.open(io.BytesIO(file.read()))

            # Preprocess the image. 
            # IMPORTANT: Change (128, 128) if your model expects a different size.
            processed_image = prepare_image(image, target_size=(96, 96))

            # Make the prediction
            prediction = model.predict(processed_image)
            
            # Interpret the prediction. A single output neuron for binary classification
            # usually gives a value between 0 and 1. We'll use 0.5 as the threshold.
            # (Assumes 'Fake' was class 0 and 'Real' was class 1 during training)
            confidence_score = prediction[0][0]
            is_fake = confidence_score < 0.5

            result_label = "Fake" if is_fake else "Real"
            
            # Calculate a user-friendly confidence percentage
            if is_fake:
                confidence_percent = (1 - confidence_score) * 100
            else:
                confidence_percent = confidence_score * 100

            return jsonify({
                'prediction': result_label,
                'confidence': f'{confidence_percent:.2f}%'
            })

        except Exception as e:
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500
    
    return jsonify({'error': 'An unknown error occurred.'}), 500

# This part is for running locally. In Colab, we'll run the app differently.
if __name__ == '__main__':
    load_app_model()
    app.run(debug=True, host="0.0.0.0", port=9129)


