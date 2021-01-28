from flask import Flask, render_template, request
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image


app = Flask(__name__)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detection', methods=['POST'])
def detection():
    canvasdata = request.form['canvasimg']
    encoded_data = request.form['canvasimg'].split(',')[1]
    print(canvasdata[:50])
    
    # Decode base64
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,255,0), 2)
    
    cv2.imwrite('frame.jpg', frame)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    pil_img = Image.fromarray(frame)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    base64_frame = base64.b64encode(buff.getvalue()).decode("utf-8")
    base64_frame = 'data:image/png;base64,' + base64_frame

    return render_template('index.html', canvasdata=base64_frame)

if __name__ == '__main__':
    app.run(debug = True)