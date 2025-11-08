import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

MODEL_PATH = "deepfake_detector_model.h5"
model = load_model(MODEL_PATH)
print(f"✅ Model '{MODEL_PATH}' loaded successfully")

def prepare_image(image, target_size=(96, 96)):
    if image is None:
        raise ValueError("No image was uploaded.")
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0
    return image_array

def predict(image):
    if image is None:
        return "<div class='text-yellow-400 font-bold'>⚠️ Please upload an image before analyzing.</div>"

    processed_image = prepare_image(image)
    prediction = model.predict(processed_image)
    confidence_score = float(prediction[0][0])
    is_fake = confidence_score < 0.5
    result_label = "Fake" if is_fake else "Real"
    confidence_percent = (1 - confidence_score if is_fake else confidence_score) * 100

    html_result = f"""
    <div class="mt-6 p-6 rounded-lg text-2xl font-bold {'bg-green-800/50 text-green-300' if result_label=='Real' else 'bg-red-800/50 text-red-300'}">
        Result: <span class="font-black">{result_label}</span><br>
        <span class="text-lg font-normal">Confidence: {confidence_percent:.2f}%</span>
    </div>
    """
    return html_result

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🎭 Deepfake Detector\nUpload an image to determine if it is real or AI-generated.")
    image_input = gr.Image(type="pil", label="Upload Image")
    output_html = gr.HTML(label="Result")
    analyze_button = gr.Button("Analyze Image", variant="primary")
    analyze_button.click(fn=predict, inputs=image_input, outputs=output_html)

demo.launch(server_name="0.0.0.0", server_port=7860)

