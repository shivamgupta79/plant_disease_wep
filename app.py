from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
import os
import requests

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Model
MODEL_PATH = "trained_model.h5"
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
else:
    print(f"❌ Model file not found at: {MODEL_PATH}")
    model = None

# Groq API Setup
GROQ_API_KEY = "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"  # Replace with your actual key
GROQ_MODEL = "llama3-70b-8192"

# Class labels
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Image prediction
def predict_image(filepath):
    if model is None:
        return "Model not loaded", 0.0

    image = tf.keras.preprocessing.image.load_img(filepath, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    input_arr = np.expand_dims(input_arr, axis=0)

    predictions = model.predict(input_arr)
    index = np.argmax(predictions)
    confidence = float(np.max(predictions))

    return class_names[index], round(confidence * 100, 2)

# Groq advice generation
def get_disease_advice(disease_label):
    prompt = f"""
    A farmer uploaded a photo of a plant showing signs of: '{disease_label}'.
    Please provide the following:
    1. A one-line description of the disease.
    2. Best pesticide or organic treatment options.
    3. Prevention tips to avoid this disease in the future.
    4. Name of the affected crop.
    Output it in simple and clear format.
    """

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            reply = response.json()['choices'][0]['message']['content']
            return reply.strip()
        else:
            return f"❌ Groq API error: {response.text}"
    except Exception as e:
        return f"❌ Exception while calling Groq API: {e}"

# Main route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            result, confidence = predict_image(filepath)
            advice = get_disease_advice(result)

            return render_template('result.html', result=result, confidence=confidence, filepath=filepath, advice=advice)

    return render_template('index.html')

# Run
if __name__ == '__main__':
    app.run(debug=True)

