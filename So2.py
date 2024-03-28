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
x_train, x_test, y_train, y_test = create_structured_input("structured-bodyPerformance.csv", True)

# -------------------------------
# So2
# -------------------------------
def So2_create_model(hp):
  # input layer
  inputs = keras.Input(x_train.shape[1],)
  
  # hidden layers
  x = keras.layers.BatchNormalization()(inputs)
  x = keras.layers.Dense(hp.Choice('neuron', values=[16,32,64]))(x)
  x = keras.layers.ReLU(max_value=hp.Choice('max_value', values=[0,10,20]))(x)
  x = keras.layers.Dense(hp.Choice('neuron', values=[16,32,64]))(x)
  x = keras.layers.ReLU(max_value=hp.Choice('max_value', values=[0,10,20]))(x)
  x = keras.layers.Dropout(hp.Choice('dropout', values=[0.0,0.2,0.5]))(x)

  # output layer
  outputs = keras.layers.Dense(4, activation='softmax')(x)  

  model = keras.Model(inputs, outputs)
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

# apply grid search
So2_model = keras_tuner.GridSearch(
  So2_create_model
  , objective='accuracy'
  , overwrite=True
)

# search
num_epochs = 50
So2_model.search(x_train, y_train, epochs=num_epochs)

# best params
get_best_params(So2_model, "neuron", "max_value", "dropout")

# best model summary
So2_best_model = get_best_model(So2_model, x_train, y_train)

# fit using the best model
So2_best_model.fit(x_train, y_train)

# distributions of classes
distributions_of_classes(y_train, y_test)

# actual and predicted y values/ confusion matrix
generate_confusion_matrix(So2_best_model, "So2_best_model", x_test, y_test)








