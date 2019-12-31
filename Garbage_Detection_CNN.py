import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
from sklearn.preprocessing import StandardScaler

PATH ="C:\\Users\\Lenovo\\Downloads\\Smart India- Garbage Detection"

data_path = os.path.join(PATH, 'dataset.npy')
label_path = os.path.join(PATH, 'labels.npy')

dataset = np.load(data_path)
labels = np.load(label_path)

print('Total Images: ',dataset.shape[0])
print('Garbage Images: ',len(labels[labels==1]))
print('Non Garbage Images: ',len(labels[labels==0]))

def standardisation(img):
    std_scaler = StandardScaler()
    img = std_scaler.fit_transform(img)
    return img

def BlurImg(img):
    blur_img = cv2.GaussianBlur(img, (3,3), 0)
    return blur_img

def resize(img, w, h):
    r_img = cv2.resize(img, (w,h),interpolation=cv2.INTER_AREA)
    return r_img

preprocessed_data = []
for i in range(dataset.shape[0]):
    image = dataset[i]
    image_normalised = standardisation(image)
    img_resized = resize(image_normalised, 120,120)
    img_blur = BlurImg(img_resized)
    preprocessed_data.append(img_blur)


preprocessed_data = np.array(preprocessed_data)
print(preprocessed_data.shape)

X_train, X_test, y_train, y_test = train_test_split(preprocessed_data, labels, test_size=0.3, random_state = 42, shuffle=True, stratify=labels)


s = f"""Total Training Samples: {X_train.shape[0]}
Train Garabage samples: {y_train[y_train == 1].shape[0]}
Train Non Garbage Samples: {y_train[y_train == 0].shape[0]}\n
Test Samples: {X_test.shape[0]}
Test Garabage samples: {y_test[y_test == 1].shape[0]}
Test Non Garbage Samples: {y_test[y_test == 0].shape[0]}"""

print(s)

X_train = X_train.reshape(X_train.shape[0],120,120,1)
X_test = X_test.reshape(X_test.shape[0],120,120,1)

num_classes = 1

def cnn_model():
    model = Sequential()
    model.add(Conv2D(60,(5,5), input_shape=(120,120,1), activation='relu'))
    model.add(Conv2D(60, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    model.add(Conv2D(30, (3,3), activation='relu'))
    model.add(Conv2D(30, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(units= 200,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation = 'sigmoid'))
    
    # Compile Model
    model.compile(Adam(lr=0.001), loss='mean_squared_error', metrics=['accuracy'])
    
    return model

model = cnn_model()
print(model.summary())

history = model.fit(X_train, y_train, batch_size=30, epochs=10, validation_split=0.2)

score = model.evaluate(X_test,y_test)

print('Test Score :',score[0],'Test Accuracy :',score[1])

plt.subplot(2,1,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['Training','Validation'])
plt.xlabel('epochs')
plt.title('Accuracy')


plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.xlabel('epochs')
plt.title('Loss')
plt.show()







