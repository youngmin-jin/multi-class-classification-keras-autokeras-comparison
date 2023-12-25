# libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner
# from input_output import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.xception import preprocess_input

# -------------------------------
# version check
# -------------------------------
import tensorflow as tf
from tensorflow import keras
import keras_tuner

print('tensorflow', tf.__version__)
print('keras', keras.__version__)
print('keras_tuner', keras_tuner.__version__)

# -------------------------------
# read and modify data
# -------------------------------
# generate a dataset
# x_train, x_test, y_train, y_test = create_image_input('Multi-class Weather Dataset',True)

# generate a dataset
data_dir = 'Multi-class Weather Dataset'
# image_size = (180, 180)
image_size=(224,224)
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
x_train, x_test, y_train, y_test = train_test_split(images_array, labels_array, train_size=0.7)

# data augmentation model
data_augmentation = keras.Sequential(
    [layers.BatchNormalization()
     , layers.RandomFlip('horizontal')
     , layers.RandomRotation(0.1)
     , layers.RandomContrast(0.1)
    ]
)

# apply data augmentation to the training dataset
x_train = data_augmentation(x_train)
# x_train = preprocess_input(x_train)

# apply one-hot encoding to target variables
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)



# -------------------------------
# i2 
# -------------------------------
def i2_create_model(hp):
#  image_size = (180, 180)
  image_size = (224,224)
  inputs = keras.Input(shape=image_size+(3,)) 

  xception_model = keras.applications.Xception(
      include_top=False
      , weights='imagenet'
      , pooling=hp.Choice("pooling", values=['avg','max'])
  )(inputs)
  x = keras.layers.Dense(hp.Choice('neuron', values=[100,500,1000,1500,2000]), activation=hp.Choice('activation', values=['relu','sigmoid','tanh']))(xception_model)
#  for i in range(hp.Int('hidden_layers', 0, 2)):
#    x = keras.layers.Dense(hp.Choice('neuron', values=[100,500,1000,1500,2000]), activation=hp.Choice('activation', values=['relu','sigmoid','tanh']))(x)
#  x = layers.Dropout(hp.Choice('dropout', values=[0.0,0.2,0.5]))(x)
  
  outputs = keras.layers.Dense(4, activation='softmax')(x)  
  model = keras.Model(inputs, outputs)
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  return model

# random search
i2_model = keras_tuner.GridSearch(
  i2_create_model
  , objective='accuracy'
  , overwrite=True
)

# search
num_epochs = 5
i2_model.search(x_train, y_train, epochs=num_epochs)

# best model results
print("---------------- best params --------------------")
i2_best_param = i2_model.get_best_hyperparameters(num_trials=1)[0]
# print("weights: ", i2_best_param.get("weights"))
print("pooling: ", i2_best_param.get("pooling"))
print("neuron: ", i2_best_param.get("neuron"))
print("activation: ", i2_best_param.get("activation"))
# print("dropout: ", i2_best_param.get("dropout"))
# print("hidden_layers: ", i2_best_param.get("hidden_layers"))

print("---------------- best model results --------------------")
i2_best_model = i2_model.get_best_models(1)[0]
i2_best_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(i2_best_model.summary())

# fit using the best model
i2_best_model.fit(x_train, y_train)

# actual and predict values
print("----------------- y_actual ----------------------")
y_actual = np.argmax(np.array(y_test), axis=1)
print(y_actual)

print("----------------- y_predict ----------------------")
y_predict = i2_best_model.predict(x_test)
y_predict = y_predict.argmax(axis=-1)
print(y_predict)

# confusion matrix
print("----------------- confusion matrix ----------------------")
print(confusion_matrix(y_actual, y_predict))

# confusion report
print("----------------- confusion report ----------------------")
print(classification_report(y_actual, y_predict))



