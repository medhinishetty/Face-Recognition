# -*- coding: utf-8 -*-
"""
Created on Mon May 25 09:49:30 2020

@author: MEDHINI
"""


from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from PIL import Image

model = load_model('C:\\Users\\MEDHINI\\Desktop\\Deep-Learning-Face-Recognition-master\\model.h5',compile=False)

# Loading the cascades
face_cascade = cv2.CascadeClassifier('C:\\Users\\MEDHINI\\Desktop\\Deep-Learning-Face-Recognition-master\\haarcascade_frontalface_default.xml')
classlabel=["kangna","priyanka"]


# def face_detector(img):
#     # Convert image to grayscale
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     faces = face_classifier.detectMultiScale(gray,1.3,5)
#     if faces is ():
#         return (0,0,0,0),np.zeros((48,48),np.uint8),img

#     for (x,y,w,h) in faces:
#         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#         roi_gray = gray[y:y+h,x:x+w]

#     try:
#         roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
#     except:
#         return (x,w,y,h),np.zeros((48,48),np.uint8),img
#     return (x,w,y,h),roi_gray,img


cap = cv2.VideoCapture(0)



while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    
    faces = face_cascade.detectMultiScale(frame,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        face= frame[y:y+h,x:x+w]
        face = cv2.resize(face,(224, 224))
    # rect,face,image = face_detector(frame)


        if np.sum([face])!=0:
            im = Image.fromarray(face, 'RGB')
           #Resizing into 128x128 because we trained the model with this image size.
            img_array = np.array(im)
                    #Our keras model used a 4D tensor, (images x height x width x channel)
                    #So changing dimension 128x128x3 into 1x128x128x3 
            img_array = np.expand_dims(img_array, axis=0)
        # make a prediction on the ROI, then lookup the c
            preds = model.predict(img_array)[0]
            label=classlabel[preds.argmax()]
            label_position = (x-10,y-5)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),2)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
