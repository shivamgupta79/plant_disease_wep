# 🌿 Plant Disease Detection Using CNN + Groq LLM Advice

A deep learning-based web application to detect plant leaf diseases from images using a trained CNN model and provide organic treatment advice using Groq's LLaMA-3 model.

---

## 📌 Table of Contents

- [🔍 Demo](#-demo)
- [📁 Project Structure](#-project-structure)
- [🧠 Model Architecture](#-model-architecture)
- [📦 Installation](#-installation)
- [▶️ How to Run](#-how-to-run)
- [🖼️ Example Output](#-example-output)
- [🌐 Web App Features](#-web-app-features)
- [📊 Training Script](#-training-script)
- [🧠 Groq LLM Integration](#-groq-llm-integration)
- [📌 Future Work](#-future-work)

---

## 🔍 Demo

![App Screenshot](static/demo-screenshot.png)

> Upload a leaf image ➜ Predict disease ➜ Get treatment & prevention tips.

---

## 📁 Project Structure
![Flow Chart](https://github.com/user-attachments/assets/be460dfd-0f2a-4ceb-b4bc-e19a984f534d)




## 🧠 Model Architecture

CNN built using `TensorFlow/Keras`:
- 5× Conv2D + MaxPooling Blocks
- Dropout Layers to prevent overfitting
- Dense Layer (1500 neurons)
- Output: 38 Classes (Softmax)

Trained for 10 epochs on 128×128 resized RGB images.

---

## 📦 Installation

### Python 3.8 or higher recommended

```bash
git clone https://github.com/yourusername/plant-disease-detector.git
cd plant-disease-detector
pip install -r requirements.txt
Or manually install:

bash
Copy
Edit
pip install flask tensorflow numpy matplotlib seaborn requests
▶️ How to Run
🔬 Train Model (Optional)
bash
Copy
Edit
python Main.py
Saves trained_model.h5 and training_hist.json.

🌐 Start Web App
bash
Copy
Edit
python app.py
Visit in browser:
http://127.0.0.1:5000/

🖼️ Example Output
After uploading an image:

yaml
Copy
Edit
✔️ Detected Disease: Tomato___Late_blight
📊 Confidence: 92.35%
🧠 Advice:
1. Description: Tomato late blight causes leaf decay and fruit rot.
2. Treatment: Use copper fungicides or neem oil.
3. Prevention: Remove infected plants and improve air circulation.
4. Crop: Tomato
🌐 Web App Features
Upload leaf image

Disease prediction using CNN model

AI-generated treatment advice using Groq LLM (LLaMA-3)

Confidence percentage and visual results

Mobile-friendly frontend (Flask + HTML)

📊 Training Script
Located in Main.py:

Loads images from train/, valid/

Builds & trains CNN model

Saves model (trained_model.h5)

Plots training vs validation accuracy

Generates confusion matrix and classification report

🧠 Groq LLM Integration
The web app uses Groq’s blazing-fast inference engine and LLaMA-3 model to generate:

Disease descriptions

Organic treatment options

Preventive tips

You can replace the API key in app.py:

python
Copy
Edit
GROQ_API_KEY = "your_groq_api_key"
Get your key from: https://console.groq.com

📌 Future Work
Deploy on Render / Replit / HuggingFace Spaces

Add data augmentation in training

Add image capture via webcam

Build mobile app frontend with Flutter / React Native

Use Transfer Learning (VGG16 / ResNet)
