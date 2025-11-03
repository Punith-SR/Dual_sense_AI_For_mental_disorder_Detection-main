from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from keras.models import load_model
import io
from PIL import Image

app = Flask(__name__)

# ============================================
# LOAD YOUR MODEL (Use relative paths!)
# ============================================
try:
    model = load_model('model_file.h5')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Load face detector (relative path!)
try:
    faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    print("✅ Face detector loaded successfully!")
except Exception as e:
    print(f"❌ Error loading face detector: {e}")

# Your emotion labels
labels_dict = {
    0: 'anxiety', 
    1: 'depressed', 
    2: 'not_depressed', 
    3: 'nostress', 
    4: 'stress'
}

current_label = ""

# ============================================
# ROUTES - Your Original Routes
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
# NEW ROUTE - Browser-Based Camera Prediction
# ============================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded', 'emotion': 'Error', 'success': False})
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded', 'emotion': 'Error', 'success': False})
        
        file = request.files['image']
        
        try:
            img = Image.open(io.BytesIO(file.read()))
            img_array = np.array(img)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            faces = faceDetect.detectMultiScale(gray, 1.3, 5)
            
            if faces is None or len(faces) == 0:
                return jsonify({
                    'emotion': 'No face detected',
                    'confidence': 0,
                    'success': False
                })
            
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            resized = cv2.resize(face_roi, (48, 48))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 48, 48, 1))
            
            result = model.predict(reshaped, verbose=0)
            
            # ✅ SOLUTION: Apply calibration to reduce not_depressed bias
            # Reduce the confidence of not_depressed and nostress
            calibrated = result[0].copy()
            
            # Reduce not_depressed bias (it wins too often)
            calibrated[2] = calibrated[2] * 0.85  # not_depressed
            calibrated[3] = calibrated[3] * 1.2   # boost nostress
            
            # Renormalize so they sum to 1
            calibrated = calibrated / np.sum(calibrated)
            
            emotion_label = np.argmax(calibrated)
            confidence = float(calibrated[emotion_label])
            current_label = labels_dict.get(emotion_label, 'Unknown')
            
            print(f"✅ Detected: {current_label} (Confidence: {confidence:.2f})")
            
            return jsonify({
                'emotion': current_label,
                'confidence': confidence,
                'success': True
            })
            
        except Exception as inner_error:
            print(f"Inner error: {inner_error}")
            return jsonify({'error': f'Processing error: {str(inner_error)}', 'emotion': 'Error', 'success': False})
            
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e), 'emotion': 'Error', 'success': False})


# ============================================
# RUN APP
# ============================================

if __name__ == '__main__':
    app.run(debug=True)
