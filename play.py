import numpy as np
import cv2
import os
import sys
import time
import tensorflow as tf
from tensorflow import keras
import random
from keras.preprocessing import image

model = tf.keras.models.load_model('sps.h5')

model_dict={0:"paper",1:"scissors",2:"stone"}

def random_gesture():
    l=["stone","paper","scissors"]
    m=random.randint(0,2)
    return l[m]


def who_won(user,a):
    
    
    if user==a:
        return "Tie"
    elif user=="stone":
        if a=="paper":
            return "Lost"
        else:
            return "Won"
    elif user=="paper":
        if a=="scissors":
            return "Lost"
        else:
            return "Won"
    elif user=="scissors":
        if a=="stone":
            return "Lost"
        else:
            return "Won"

    



cap = cv2.VideoCapture(0)
state="wait"
comp_score=0
score=0
text="press 'a' to play"
standing=""
while(True):
    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    frame=cv2.resize(frame,(1200,700))
    cv2.rectangle(frame, (600, 100), (1050, 550), (255, 255, 255), 2)

    cv2.putText(frame,text,(150,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),1)
    cv2.putText(frame,"Your score = "+str(score) ,(100,300),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),1)
    cv2.putText(frame,"Computer score = "+str(comp_score) ,(100,500),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),1)
    cv2.putText(frame,standing,(750,80),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),1)
    cv2.imshow('Stone-Paper-Scissors!',frame)
    

   
    if state=="play":
        img = frame[100:550, 600:1050]
        
        img=cv2.resize(img,(150,150))        
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict_classes(images)
        print(classes)
        user_input=model_dict.get(classes[0])
        a=random_gesture()
        print("ur="+user_input)
        print(a)
        output=who_won(user_input,a)
        print(output)
        text2="Computer choose :"+a
        cv2.putText(frame,text2,(750,570),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),1)
        
        if output=="Won":
            score=score+1
            standing="You Won!"
        elif output=="Lost":
            comp_score=comp_score+1
            standing="You Lost"
        else:
            standing="Match Tied"
        state="wait"

        
            

            
    
    if score>10:
        cv2.putText(frame,"YOU WON THE MATCH!",(750,200),cv2.FONT_HERSHEY_DUPLEX,2,(255,255,0),1)
        
        break
    if comp_score>10:
        cv2.putText(frame,"YOU LOST THE MATCH!",(750,200),cv2.FONT_HERSHEY_DUPLEX,2,(255,255,0),1)
        
        break

        


    k = cv2.waitKey(10)
    if k == ord('a'):
        state = "play"
    if k== ord('q') :
        break

cap.release()
cv2.destroyAllWindows()