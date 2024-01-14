# libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import keras_tuner
from input_output import *
from tensorflow.keras.applications.xception import preprocess_input

# -------------------------------
# version check 
# -------------------------------
print('tensorflow', tf.__version__)
print('keras', keras.__version__)
print('keras_tuner', keras_tuner.__version__)

# -------------------------------
# read the data
# -------------------------------
x_train, x_test, y_train, y_test = create_image_input('Multi-class Weather Dataset',True)

# -------------------------------
# Io2 
# -------------------------------
def Io2_create_model(hp):
  # image_size = (224,224)
  
  # input layer
  inputs = keras.Input(shape=image_size+(3,)) 

  # hidden layers
  x = keras.layers.BatchNormalization()(inputs)
  # x = keras.layers.RandomFlip('horizontal')(x)
  # x = keras.layers.RandomContrast(0.1)(x)
  x = keras.layers.Resizing(224,224)(x)
  x = keras.applications.Xception(
      include_top=False,
      weights='imagenet',
      pooling=hp.Choice("pooling", values=['avg', 'max'])
  )(x)
  # x = keras.layers.GlobalAveragePooling2D()(x)
  # x = keras.layers.Dense(200, activation=hp.Choice("activation", values=['relu','sigmoid']))(x)

  # output layer
  outputs = keras.layers.Dense(4, activation='softmax')(x)  

  model = keras.Model(inputs, outputs)
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

# apply grid search
Io2_model = keras_tuner.GridSearch(
  Io2_create_model
  , objective='accuracy'
  , overwrite=True
  , max_trials=2
)

# search
num_epochs = 3
Io2_model.search(x_train, y_train, epochs=num_epochs)

# best parameters
get_best_params(Io2_model, "pooling")

# best model summary 
Io2_best_model = get_best_model(Io2_model, x_train, y_train)

# fit using the best model
Io2_best_model.fit(x_train, y_train)

# distributions of classes
distributions_of_classes(y_train, y_test)

# actual and predicted y values/ confusion matrix
generate_confusion_matrix(Io2_best_model, "Io2_best_model", x_test, y_test)



