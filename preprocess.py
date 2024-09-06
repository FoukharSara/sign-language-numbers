import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers,models

data_dir= 'dataset'
sign_labels = {
    0: 'zero',
    1: 'one',
    2: 'two',
    3: 'three',
    4: 'four',
    5: 'five',
    6:'six',
    7:'seven',
    8:'eight',
    9:'nine'    
}
def to_int(x):
    return int(x)

categories = sorted(os.listdir(data_dir),key = to_int)

def load_data(data_dir,categories):
    data=[]
    labels=[]
    for categorie in categories:
        path = os.path.join(data_dir,categorie)
        label = int(categorie)  
        for img in os.listdir(path):
            img_path=os.path.join(path,img)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (64, 64),interpolation=cv2.INTER_CUBIC)
            data.append(image)
            labels.append(label)
    data=np.array(data).reshape(-1,64,64,1)
    data = data/255.0
    labels = np.array(labels)
    return data,labels

data,labels = load_data(data_dir,categories)
X_train,X_test,Y_train,Y_test=train_test_split(data,labels,test_size=0.2,random_state=42)

model = models.Sequential([
    layers.Conv2D(128,(3,3), activation='relu',input_shape=(64,64,1)),
    layers.MaxPooling2D(),
    
    layers.Conv2D(64,(3,3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(32,(3,3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(16,(3,3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(X_train,Y_train,epochs=10, validation_data=(X_test, Y_test))

test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print(f'Test accuracy: {test_accuracy}')
model.save('model/emotion_detector_model.h5')
np.save("data/X_train.npy",X_train)
np.save("data/X_test.npy",X_test) 
np.save("data/Y_train.npy",Y_train) 
np.save("data/Y_test.npy",Y_test)
