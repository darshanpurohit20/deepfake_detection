## Deepfake Image Detection System

<img src="https://github.com/user-attachments/assets/ff9ce3f0-176a-4bcb-bb5c-36ac09254310" alt="Deepfake Detector Banner" width="180" align="left" style="margin-right: 15px;"/>

An AI-powered system that detects whether an uploaded image is **real or AI-generated (deepfake)** using a **TensorFlow CNN model**.  
Trained, saved, Dockerized, and deployed seamlessly on **Hugging Face Spaces**.  

This project showcases the fusion of **deep learning and explainable AI** to combat misinformation in digital media.  
Built with a focus on **accuracy, scalability, and real-world deployment**, it serves as a foundation for advanced multimodal deepfake detection.

<br clear="left"/>


<img width="7236" height="4540" alt="image" src="https://github.com/user-attachments/assets/7ebe634e-cfdb-4402-8bce-fd293f905cd2" />

### 🧠 Overview  
**Deepfake Image Detection System** is an AI-powered tool designed to identify **AI-generated (fake)** and **authentic (real)** images using a deep learning model trained with TensorFlow.  
This project demonstrates an end-to-end workflow — **from model training and saving**-->done in colab, to **Dockerized deployment** on **Hugging Face Spaces**.

🔗 **Live Demo:** [Deepfake-Audio on Hugging Face / by Darshan Purohit](https://huggingface.co/spaces/Darshanpurohit/deepfake-audio)

---

## 🚀 Features
- 🧩 Detects **real vs deepfake** images with a confidence score  
- ⚡ Powered by a **TensorFlow CNN** trained on curated image datasets  
- 💾 Model serialized and loaded from `deepfake_detector_model.h5`  
- 🐳 Fully **Dockerized** for portability and reproducible builds  
- ☁️ Deployed seamlessly to **Hugging Face Spaces** with Gradio UI  

---

## 🧰 Tech Stack
| Component | Technology |
|------------|-------------|
| **Frontend / UI** | Gradio (interactive image upload interface) |
| **Backend / Model** | TensorFlow + Keras |
| **Deployment** | Docker + Hugging Face Spaces |
| **Language** | Python|
| **Environment** | `python:3.13.3-slim` |

---

## ⚙️ How It Works
1. Upload an image (PNG/JPG).  
2. The image is resized and normalized to `(96, 96)` pixels.  
3. The model predicts the likelihood of it being **fake or real**.  
4. The app displays a confidence score and visual feedback.  

---

## 🐳 Docker Setup

### Dockerfile
```dockerfile
FROM python:3.13.3-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 7860
CMD ["python", "app.py"]
```
Build & Run Locally 

# 1)Run with Docker

```bash
# Start Docker Desktop
open -a Docker

# Check if Docker is running
docker info

# Build your Docker image
docker build -t deepfake-detector .

# Run the container and expose Gradio’s default port
docker run -p 7860:7860 deepfake-detector

```

# 2)Run Locally (using Virtual Environment)
```
#Step 1: Create a virtual environment
# macOS / Linux
python3 -m venv venv

# Windows
python -m venv venv

# Step 2: Activate the environment
# macOS / Linux
source venv/bin/activate

# Windows (Command Prompt)
venv\Scripts\activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run the app
python app.py
```
## Model Information
- **Model Type:** Convolutional Neural Network (CNN)  
- **Framework:** TensorFlow / Keras  
- **Saved Format:** `deepfake_detector_model.h5`  
- **Input Shape:** `96 × 96 × 3 (RGB)`  
- **Output:** Binary Classification → **Real / Fake**

---

## Deployment
This project is fully containerized using **Docker** and deployed on **Hugging Face Spaces** for public access and reproducibility.

🔗 **Live Demo:** [Darshanpurohit / Deepfake-Audio](https://huggingface.co/spaces/Darshanpurohit/deepfake-audio)

---

## 🧑‍💻 Author
**Darshan Purohit**  
🚀 AI & Deep Learning Enthusiast • Data Science Explorer  
📂 **GitHub:** [github.com/darshanpurohit20](https://github.com/darshanpurohit20)  
🌍 Passionate about building practical AI tools for real-world problems.

---

## 🪄 Future Improvements
- 🎬 Add **multimodal detection** for video and audio-based deepfakes  
- 🔍 Integrate **Grad-CAM explainability** for visual insights  
- ⚡ Optimize model performance for **real-time inference**

---

## 🏁 License
Licensed under the **MIT License** — free for educational and research use.  
You are welcome to fork, modify, and build upon this project with attribution.

---

⭐ **If you found this project helpful, please give it a star!**
