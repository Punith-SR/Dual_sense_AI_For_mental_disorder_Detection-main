
# DUAL SENSE AI FOR MENTAL DISORDER DETECTION

Detecting the mental disorders like stress,nostress,depression,notdepressed and anxiety using facial expression and speech.




## Installation

install Tensorflow: pip install Tensorflow

install Keras: pip install Keras

install Flask: pip install Flask

install speechrecognition: pip install SpeechRecognition




## Requirements

1. use of internet is mandatory.

2. Camera with appropriate light facility

3. Microphone
## Instructions to run project
1. change the path in mains.py 
load_model("path of model_file.h5") 
faceDetect("path of haarcascade_frontalface_default.xml)

2. change the path in app.py(to connect main.py)
def run_face_detection("path of main.py")

3. run app.py in vs code  as "python app.py"

4. Click on generated link.
* click on start button for next page.
* click on  disorder buttons for quick guideline 

5. Face-based Detection
click on start face detection 
*The camera is activated for 1min for detecting  the face emotions.If required to stop early click on "q" button in keyboard.

Speech-Based Analysis
upload the text_data.csv from respective folder of your system.
click on start speech recognition and provide the sentence containing the words in text_data.csv

6. Combine result
results will appear here once both methods are processed.

##example to support the above analysis  
face-based analysis: stress  
speech-based analysis: stress  
combined result: stress

faced-based analsis: stress  
voice-based analysis: nostress  
combine result: result not matched 


