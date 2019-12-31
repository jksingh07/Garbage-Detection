from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import os
import cv2
import numpy as np
##import matplotlib.pyplot as plt


model_used = 'CNN'
#model_used = 'RF'

if model_used == 'CNN':
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers.convolutional import Conv2D, MaxPooling2D
    from keras.optimizers import Adam

    model = joblib.load('CNN_model.pkl')


elif model_used == 'RF':
    model = joblib.load('RF_model_1.pkl')

std = StandardScaler()
categories = ['Non Garbage', 'Garbage']
a=0

while(True):
    url='http://192.168.0.127:8080/shot.jpg'
    cap = cv2.VideoCapture(url)
##    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 60)
##    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 20)
    ret, frame = cap.read()
    test_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    std_img = std.fit_transform(test_img)
    std_img = cv2.resize(std_img, (120,120),interpolation=cv2.INTER_AREA)
    blur_img = cv2.GaussianBlur(std_img, (3,3),0)

    if model_used == "CNN":
        image = np.reshape(std_img,(1,120,120,1))
    elif model_used == 'RF':
        image = std_img.ravel()
        image = image.reshape(1,-1)
        
    prediction = model.predict(image)
    print('Probability of Garbage: ',prediction[0])
    result = categories[int(np.round_(prediction[0]))]

    black = [0,0,0]     #---Color of the border---
    constant=cv2.copyMakeBorder(frame,10,10,10,10,cv2.BORDER_CONSTANT,value=black )
    #--- Here I created a violet background to include the text ---
    violet= np.zeros((100, constant.shape[1], 3), np.uint8)
    violet[:] = (255, 0, 180)
    vcat = cv2.vconcat((violet, constant))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vcat,result,(30,50), font, 2,(0,0,0), 3, 0)

    print(frame)

    if frame is not None:
        #while a>20:
        cv2.imshow('Garbage Detection', vcat)
            #plt.imshow(frame)
        #    a=0
       # a+=1
    q = cv2.waitKey(1)
    if q == ord('q'):
        break

cv2.destroyAllWindows()



    
