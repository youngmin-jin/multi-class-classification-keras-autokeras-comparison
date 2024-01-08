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
# So 
# -------------------------------
# model
def So_create_model(hp):
  model = keras.models.Sequential()
  model.add(keras.layers.Dense(hp.Choice('neuron', values=[100,500,1000,1500,2000])
  , input_shape=(x_train.shape[1],)
  , activation=hp.Choice('activation', values=['relu','sigmoid','tanh'])))
  for i in range(hp.Int('hidden_layers', 2, 4)):
    model.add(keras.layers.Dense(hp.Choice('neuron', values=[100,500,1000,1500,2000])
    , activation=hp.Choice('activation', values=['relu','sigmoid','tanh'])))
  model.add(keras.layers.Dropout(hp.Choice('dropout', values=[0.0,0.2,0.5])))
  model.add(keras.layers.Dense(4, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

# apply grid search
So_model = keras_tuner.GridSearch(
  So_create_model
  , objective='accuracy'
  , overwrite=True
)

# search
num_epochs = 3
So_model.search(x_train, y_train, epochs=num_epochs)

# best params
get_best_params(So_model, "hidden_layers", "neuron", "activation", "dropout")

# best model summary
So_best_model = get_best_model(So_model, x_train, y_train)

# fit using the best model
So_best_model.fit(x_train, y_train)

# distributions of classes
distributions_of_classes(y_train, y_test)

# actual and predicted y values/ confusion matrix
generate_confusion_matrix(So_best_model, "So_best_model", x_test, y_test)








