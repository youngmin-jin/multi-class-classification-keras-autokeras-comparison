# libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img


# -------------------------------
# version check
# -------------------------------
import tensorflow as tf
from tensorflow import keras

print('tensorflow', tf.__version__)
print('keras', keras.__version__)

# -------------------------------
# read and modify data
# -------------------------------
# generate a dataset
data_dir = 'Multi-class Weather Dataset'
image_size = (180, 180)
images = []
labels = []
class_names = os.listdir(data_dir)

# iterate data implementation using class name
for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)

    for image_file in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_file)
        image = load_img(image_path)
        
        # resize data to form the same shape/ convert them into array
        image = image.resize(image_size)              
        image = img_to_array(image)
        
        # append to the array
        images.append(image)        
        labels.append(class_name)

# convert list into array
images_array = np.stack(images)
labels_array = np.array(labels)

# split
x_train, x_test, y_train, y_test = train_test_split(images_array, labels_array, test_size=0.3)

# data augmentation model
data_augmentation = keras.Sequential(
    [layers.RandomFlip('horizontal')
      , layers.RandomRotation(0.1)
    ]
)

# apply data augmentation to the training dataset
x_train = data_augmentation(x_train)

# apply one-hot encoding to target variables
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)


# -------------------------------
# i1
# -------------------------------
inputs = keras.Input(shape=image_size+(3,))

x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(inputs)
x = layers.SeparableConv2D(128, 3, padding="same", activation='relu')(x)
x = layers.GlobalMaxPooling2D()(x)
x = layers.Dropout(0.5)(x)

# output layer
outputs = layers.Dense(4, activation='softmax')(x)

# model
i1_model = keras.Model(inputs, outputs)

# compile
i1_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# fit
num_epochs = 100
i1_model.fit(x_train, y_train, epochs=num_epochs)

# summary
print('----------- results ---------------')
print(i1_model.summary())

# actual and predict values
print("----------------- y_actual ----------------------")
y_actual = np.argmax(np.array(y_test), axis=1)
print(y_actual)

print("----------------- y_predict ----------------------")
y_predict = i1_model.predict(x_test)
y_predict = y_predict.argmax(axis=-1)
print(y_predict)

# confusion matrix
print("----------------- confusion matrix ----------------------")
print(confusion_matrix(y_actual, y_predict))

# confusion report
print("----------------- confusion report ----------------------")
print(classification_report(y_actual, y_predict))


