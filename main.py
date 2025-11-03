import cv2
import numpy as np
from keras.models import load_model
import time
import requests  # Import for HTTP requests

# Load the model
model = load_model(
    r"C:\2 sudarsh\Dual_sense_AI_For_mental_disorder_Detection-main\model_file.h5"
)

# Initialize webcam
#video = cv2.VideoCapture(0)

# Initialize face detector
faceDetect = cv2.CascadeClassifier(
    r"C:\2 sudarsh\Dual_sense_AI_For_mental_disorder_Detection-main\haarcascade_frontalface_default.xml"
)


# Emotion labels (same as in main.py)
labels_dict = {0: 'anxiety', 1: 'depressed', 2: 'not_depressed', 3: 'nostress', 4: 'stress'}

# Set the timer duration (in seconds)
timer_duration = 60  # 1 minute
start_time = time.time()  # Get the start time

# Flask server URL to send the detected label
SERVER_URL = "http://127.0.0.1:5000/update-label"  # Flask endpoint

while True:
    ret, frame = video.read()
    
    # Check if 1 minute has passed
    if time.time() - start_time > timer_duration:
        print("Timer finished. Exiting...")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)
    
    for x, y, w, h in faces:
        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)
        
        label = np.argmax(result, axis=1)[0]
        
        # Ensure label is within valid range
        if label not in labels_dict:
            print(f"Invalid label: {label}, skipping this frame.")
            continue  # Skip this frame if label is invalid
        
        # Draw bounding box and label on the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Send the detected label to Flask server
        requests.post(SERVER_URL, json={"label": labels_dict[label]})

    cv2.imshow("Frame", frame)
    
    # Wait for a key press, exit if 'q' is pressed
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
