import numpy as np
import cv2
import os
import sys

labels=["paper","stone","scissors"]

samples=(input("enter number of samples for each=    "))

for label in labels:
    count=0
    IMG_SAVE_PATH = 'image_data'
    IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, label)

    try:
        os.mkdir(IMG_SAVE_PATH)
    except FileExistsError:
        pass
    try:
        os.mkdir(IMG_CLASS_PATH)
    except FileExistsError:
        print("{} directory already exists.".format(IMG_CLASS_PATH))
        print("All images gathered will be saved along with existing items in this folder")

    cap = cv2.VideoCapture(0)
    start=False


    while(True):
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        frame=cv2.resize(frame,(1200,700))
        cv2.rectangle(frame, (600, 100), (1050, 550), (255, 255, 255), 2)

        cv2.putText(frame,"Collecting images for "+label,(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),1)
        cv2.putText(frame,"Count = "+str(count),(750,80),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),1)
        cv2.imshow('Collecting images!',frame)
        

        if count>int(samples):
            break
        if start:
            img = frame[100:550, 600:1050]
            
            save_path = os.path.join(IMG_CLASS_PATH, (str(count+1)+'.jpg'))
            cv2.imwrite(save_path, img)
            count += 1        
        k = cv2.waitKey(10)
        if k == ord('a'):
            start = True
        if k== ord('q') :
            break

    cap.release()
    cv2.destroyAllWindows()