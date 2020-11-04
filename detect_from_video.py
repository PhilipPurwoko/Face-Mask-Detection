import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Write text on video
def blit(text,frame,color,position=(100, 100)):
    cv2.putText(frame,text,position,cv2.FONT_HERSHEY_SIMPLEX,2,color,4,cv2.LINE_4)

# Load saved model
model = keras.models.load_model('face_mask_ai.h5')

# Input video file name
input_video = 'video.mp4'

# Video information
videoCapture = cv2.VideoCapture(input_video)
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Write video to .avi by datetime
today = str(datetime.datetime.today()).split(' ')
today_time = '-'.join(today[1].split(':')[:-1])

video_name = f'output-{today[0]}-{today_time}.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter(video_name,fourcc, fps, size)

print(f'Video files name : {video_name}')
print(f'Video fps : {fps}')
print(f'Video size : {size}')
print('Program is working. Please be patient...')

# Running opencv frame by frame
first_time = True
while True:
    res,frame = videoCapture.read()
    if res:
        img = cv2.resize(frame, (224,224), interpolation = cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        prediction = model.predict(np.expand_dims(img,axis=0))

        blit(f'Prediction Probability',frame,(255,255,255),(100,100))
        blit(f'Wearing mask : {prediction[0][0]}',frame,(0, 255, 0),(100,200))
        blit(f'Not wearing mask : {prediction[0][1]}',frame,(0, 0, 255),(100,300))

        if first_time:
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.axis(False)
            plt.title('First Video Frame')
            plt.show()
            first_time = False
        
        # Write detected frame by frame
        videoWriter.write(frame)
    else:
        break

videoCapture.release()
videoWriter.release()
cv2.destroyAllWindows()

print(f'Finished. Video exported to {video_name}')