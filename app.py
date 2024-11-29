from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os
import gdown
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Define model paths
model_path = 'model.h5'
svm_model_path = 'svm_model.pkl'
scaler_path = 'scaler.pkl'
class_names_path = 'class_names.pkl'

# Check if the model exists, if not, download it
def download_model(file_id, model_file_path):
    if not os.path.exists(model_file_path):
        print(f"Downloading model from Google Drive: {file_id}")
        url = f'https://drive.google.com/uc?export=download&id={file_id}'
        gdown.download(url, model_file_path, quiet=False)

# Download models if not already downloaded
download_model('19mYpbOP2ilDRfNBD40slpcx3pospZ2R0', model_path)
download_model('1ECfwi3GWfcL_ZX6K2Q9HPcIGuhxG1Bz1', svm_model_path)
download_model('1PvxeG14gOJgmLPZB5UsHGx7t3fkR2TFs', scaler_path)
download_model('19bcZ_N8Pz8vN6s8bK9pR5kLL_7Ii40x7', class_names_path)


# Load Models and Scaler
cnn_model = load_model('model.h5', compile=False)
with open('svm_model.pkl', 'rb') as svm_file:
    svm_model = pickle.load(svm_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
with open('class_names.pkl', 'rb') as class_names_file:
    class_names = pickle.load(class_names_file)


# Preprocessing function
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        # Preprocess the image and extract features
        img_array = preprocess_image(filepath)
        features = cnn_model.predict(img_array).flatten()

        # Scale the features and make predictions
        scaled_features = scaler.transform([features])
        prediction = svm_model.predict(scaled_features)[0]
        confidence = svm_model.predict_proba(scaled_features)[0][prediction] * 100

        predicted_class = class_names[prediction]

        return render_template(
            'index.html',
            image_url=f'/{filepath}',
            predicted_class=predicted_class,
            confidence=f"{confidence:.2f}%"
        )

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
