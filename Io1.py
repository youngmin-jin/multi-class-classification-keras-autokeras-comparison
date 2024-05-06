# libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import keras_tuner
from input_output import *

# -------------------------------
# version check
# -------------------------------
print('tensorflow', tf.__version__)
print('keras', keras.__version__)
print('keras_tuner', keras_tuner.__version__)

# -------------------------------
# read the data
# -------------------------------
x_train, x_test, y_train, y_test = create_image_input('Multi-class Weather Dataset', True)

# -------------------------------
# Io1 
# -------------------------------
# model
def Io1_create_model(hp):
  # input layer
  inputs = keras.Input(shape=image_size+(3,))       
  
  # hidden layers
  x = keras.layers.Conv2D(hp.Choice('neuron', values=[100,500,1000,1500,2000]), 3, strides=2, padding='same', activation=hp.Choice('activation', values=['relu','sigmoid','tanh']))(inputs)
  for i in range(hp.Int('hidden_layers', 0, 2)):
    x = keras.layers.Conv2D(hp.Choice('neuron', values=[100,500,1000,1500,2000])
        , 3, strides=2, padding='same'
        , activation=hp.Choice('activation', values=['relu','sigmoid','tanh']))(x)
  x = keras.layers.SeparableConv2D(hp.Choice('neuron', values=[100,500,1000,1500,2000]), 3, padding="same", activation=hp.Choice('activation', values=['relu','sigmoid','tanh']))(x)
  x = keras.layers.GlobalMaxPooling2D()(x)
  x = keras.layers.Dropout(hp.Choice('dropout', values=[0.0,0.2,0.5]))(x)
   
  # output layer
  outputs = keras.layers.Dense(4, activation='softmax')(x)  
  
  model = keras.Model(inputs, outputs)
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

# apply grid search
Io1_model = keras_tuner.GridSearch(
  Io1_create_model
  , objective='accuracy'
  , overwrite=True
)

# search
num_epochs = 100
Io1_model.search(x_train, y_train, epochs=num_epochs)

# best parameters
get_best_params(Io1_model, "neuron", "hidden_layers", "activation", "dropout")

# best model summary
Io1_best_model = get_best_model(Io1_model, x_train, y_train)

# fit using the best model
Io1_best_model.fit(x_train, y_train)

# distributions of classes
distributions_of_classes(y_train, y_test)

# actual and predicted y values/ confusion matrix
generate_confusion_matrix(Io1_best_model, "Io1_best_model", x_test, y_test)

