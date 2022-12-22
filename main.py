import cv2
import numpy as np
import time
from cnn_model import *
from dataset import *
from train_cnn import *
import os 

# val_data = get_data_cnn('val.json')

cktp_name = 'params'
checkpoint_filepath = os.path.join('cktp', f'{cktp_name}.h5')
model = my_model((224,224,1),2)
model.load_weights(checkpoint_filepath)
print("Load Weight DONE !!!")

# eval(model, val_data)

def predict_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (224, 224))
    pred_img = np.expand_dims(img, axis=0)
    result = model.predict(pred_img)
    result = np.argmax(result, axis=-1)[0]


    cv2.putText(img, str(result), (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow('crop', img)

    return img



backSub = cv2.createBackgroundSubtractorMOG2()

capture = cv2.VideoCapture('Liverpool - 46102.mp4')

new_time = time.time()
old_time = time.time()

while True:
    _, frame = capture.read()
    if not _:
        break
    
    fgMask = backSub.apply(frame)
    fgMask = cv2.cvtColor(fgMask, 0)

    kernel = np.ones((5,5), np.uint8)
    fgMask = cv2.erode(fgMask, kernel, iterations=1) 
    fgMask = cv2.dilate(fgMask, kernel, iterations=1)
    fgMask = cv2.GaussianBlur(fgMask, (3,3), 0)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
    _,fgMask = cv2.threshold(fgMask,130,255,cv2.THRESH_BINARY)

    fgMask = cv2.Canny(fgMask,20,200)
    contours,_ = cv2.findContours(fgMask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for i in range(len(contours)):
        (x, y, w, h) = cv2.boundingRect(contours[i])
        area = cv2.contourArea(contours[i])
        if area > 4500:
            cv2.drawContours(fgMask, contours[i], 0, (0, 0, 255), 6)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)
            result = predict_img(frame[y:y+h, x:x+w])


    new_time = time.time()
    fps = int(1/(new_time - old_time))
    old_time = new_time

    cv2.putText(frame, str(fps), (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5, cv2.LINE_AA)

    frame = cv2.resize(frame, (640, 360))
    cv2.imshow('Frame', frame)
    
    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
