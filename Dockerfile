# ==============================
# 🐳 Deepfake Detector Dockerfile
# ==============================

# 1️⃣ Use a stable base image
FROM python:3.13.3-slim


# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Copy all project files into the container
COPY . /app

# 4️⃣ Install dependencies
# Using --no-cache-dir keeps the image smaller
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


# 5️⃣ Expose Gradio’s default port
EXPOSE 7860

# 6️⃣ Run the app
CMD ["python", "app.py"]
