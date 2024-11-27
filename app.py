from flask import Flask, render_template, request, jsonify
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pickle
import os
from tensorflow.config import experimental

# Disable GPU and force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load pre-trained models and data
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

with open('models/svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)


def extract_features(img_path):
    """Extract features using VGG16 model."""
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    features = vgg16_model.predict(img_array)
    return features.flatten()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        # Extract features and make predictions
        features = extract_features(filepath)
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
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
