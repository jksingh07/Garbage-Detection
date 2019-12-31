import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


model = joblib.load('CNN_model.pkl')

#print(model.summary())

path ="C:\\Users\\Lenovo\\Downloads\\Smart India- Garbage Detection\\ambiguous-annotated-images"
categories = ['Non Garbage', 'Garbage']

test_images = []
for i in os.listdir(path):
    path1 = os.path.join(path,i)
    for j in os.listdir(path1):
        img_path = os.path.join(path1, j)
        test_images.append(img_path)

std = StandardScaler()
#print(test_images)

img_idx = 7

org_img = cv2.imread(test_images[img_idx])
test_img = cv2.imread(test_images[img_idx],0)
std_img = std.fit_transform(test_img)

print(std_img.shape)
std_img = cv2.resize(std_img, (120,120),interpolation=cv2.INTER_AREA)
blur_img = cv2.GaussianBlur(std_img, (3,3),0)
print(std_img.shape)

image = np.reshape(std_img,(1,120,120,1))
prediction = model.predict(image)
print('Probability of Garbage: ',prediction[0])
plt.title(categories[int(np.round_(prediction[0]))])
plt.imshow(org_img)
plt.show()













