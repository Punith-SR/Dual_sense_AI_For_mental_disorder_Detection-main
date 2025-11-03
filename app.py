from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import io
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ============================================
# LOAD MODEL - LIGHTWEIGHT VERSION
# ============================================
model = None

def load_model_safe():
    global model
    try:
        # Try direct keras.saving first (lightweight!)
        from keras import saving
        model = saving.load_model('model.keras')
        print("✅ Model loaded (Keras format)!")
        return
    except:
        pass
    
    try:
        # Try TensorFlow fallback
        import tensorflow as tf
        model = tf.keras.models.load_model('model.keras')
        print("✅ Model loaded (TensorFlow)!")
        return
    except:
        pass
    
    try:
        # Last resort: try .h5
        from keras.models import load_model
        model = load_model('model_file.h5')
        print("✅ Model loaded (.h5 format)!")
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        model = None

# Load face detector
faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

labels_dict = {0: 'anxiety', 1: 'depressed', 2: 'not_depressed', 3: 'nostress', 4: 'stress'}
current_label = ""

# ============================================
# ROUTES
# ============================================

@app.route('/')
def start_page():
    return render_template('start.html')

@app.route('/second')
def second_page():
    return render_template('second.html')

@app.route('/main')
def index():
    return render_template('index.html')

@app.route('/get-label', methods=['GET'])
def get_label():
    global current_label
    return jsonify({"label": current_label})

@app.route('/update-label', methods=['POST'])
def update_label():
    global current_label
    data = request.json
    current_label = data.get("label", "")
    return jsonify({"status": "Label updated!"})

# ============================================
# PREDICT - EMOTION DETECTION
# ============================================

@app.route('/predict', methods=['POST'])
def predict():
    global model
    
    try:
        if model is None:
            load_model_safe()
        
        if model is None:
            return jsonify({'error': 'Model not loaded', 'emotion': 'Error', 'success': False})
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image', 'emotion': 'Error', 'success': False})
        
        file = request.files['image']
        
        try:
            img = Image.open(io.BytesIO(file.read()))
            img_array = np.array(img)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            faces = faceDetect.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return jsonify({'emotion': 'No face', 'confidence': 0, 'success': False})
            
            x, y, w, h = faces[0]
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48)) / 255.0
            face = np.reshape(face, (1, 48, 48, 1))
            
            pred = model.predict(face, verbose=0)[0]
            
            # Calibration
            pred[2] *= 0.85
            pred[3] *= 1.2
            pred /= pred.sum()
            
            emotion = labels_dict[np.argmax(pred)]
            conf = float(pred.max())
            
            return jsonify({'emotion': emotion, 'confidence': conf, 'success': True})
            
        except Exception as e:
            return jsonify({'error': str(e), 'emotion': 'Error', 'success': False})
            
    except Exception as e:
        return jsonify({'error': str(e), 'emotion': 'Error', 'success': False})

# ============================================
# RUN
# ============================================

if __name__ == '__main__':
    load_model_safe()
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
