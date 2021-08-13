# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 01:08:23 2021

@author: smith
"""
import torch 
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt


face_classifier = cv2.CascadeClassifier("A:/DOWNLOADS/haarcascade_frontalface_default.xml")

model_state = torch.load("A:/BE/HAR/resnet.pt")

data_dir = r'C:\Users\smith\Videos\UCF101\AutoEncoder-Video-Classification-master'
class_labels = os.listdir(data_dir + "/test")
#%%

import architecture 
from architecture import architecture
model=architecture.resnet18(19)

model.load_state_dict(model_state)
#%%
cap = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        
        roi_gray = cv2.resize( frame,(224, 224))

        if np.sum([roi_gray]) != 0:
            roi = tt.functional.to_pil_image(frame)
           # roi = tt.functional.to_grayscale(roi)
            roi = tt.ToTensor()(roi).unsqueeze(0)

            # make a prediction on the ROI
            tensor = model(roi)
            print(tensor)
            pred = torch.max(tensor, dim=1)[1].tolist()
            label = class_labels[pred[0]]
            
            label_position = (x, y)
            cv2.putText(
                frame,
                label,
                label_position,
                cv2.FONT_HERSHEY_COMPLEX,
                2,
                (0, 255, 0),
                3,
            )
        else:
            cv2.putText(
                frame,
                "No Face Found",
                (20, 60),
                cv2.FONT_HERSHEY_COMPLEX,
                2,
                (0, 255, 0),
                3,
            )

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
























#%%
# =============================================================================
# cap = cv2.VideoCapture(0)
# 
# 
# # import the opencv library
# import cv2
#   
#   
# # define a video capture object
# vid = cv2.VideoCapture(r'C:\Users\smith\Videos\UCF101\AutoEncoder-Video-Classification-master\UCF\Typing\v_Typing_g01_c02.avi')
#   
# while(True):
#       
#     # Capture the video frame
#     # by frame
#     ret, frame = vid.read()
#     roi=cv2.resize(frame, (244, 244))
#     roi = tt.ToTensor()(roi).unsqueeze(0)
#     tensor=model(roi)
#     pred = torch.max(tensor, dim=1)[1].tolist()
#     label = class_labels[pred[0]]
#     print(label)
#   
#     # Display the resulting frame
#     cv2.imshow('frame', frame)
#       
#     # the 'q' button is set as the
#     # quitting button you may use any
#     # desired button of your choice
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#   
# # After the loop release the cap object
# vid.release()
# # Destroy all the windows
# cv2.destroyAllWindows()
# =============================================================================
