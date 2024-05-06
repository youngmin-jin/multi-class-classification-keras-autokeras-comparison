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
y_train, y_test, train_ds, test_ds = create_text_input('text-FinancialSentimentAnalysis.csv', True, False)

# -------------------------------
# To1
# -------------------------------
def To1_create_model(hp):
  # input layer
  inputs = keras.Input(shape=(None,), dtype='int64')
  
  # embedding layer
  x = keras.layers.Embedding(max_features, embedding_dim)(inputs) 
  
  # hidden layers
  for i in range(hp.Int('hidden_layers', 1, 3)):
    x = keras.layers.Conv1D(hp.Choice('neuron', values=[100,500,1000,1500,2000])
    , 3, strides=2, padding='same'
    , activation=hp.Choice('activation', values=['relu','sigmoid','tanh']))(x)
  x = keras.layers.GlobalMaxPooling1D()(x)
  x = keras.layers.Dropout(hp.Choice('dropout', values=[0.0,0.2,0.5]))(x)
  
  # output layer
  outputs = keras.layers.Dense(3, activation='softmax', name='predictions')(x)
  
  model = keras.Model(inputs, outputs)
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

# apply grid search
To1_model = keras_tuner.GridSearch(
  To1_create_model
  , objective='accuracy'
  , overwrite=True
)

# search
num_epochs = 100
To1_model.search(train_ds, epochs=num_epochs)

# best parameters 
get_best_params(To1_model, "hidden_layers", "neuron", "activation", "dropout")

# best model summary
To1_best_model = get_best_model(To1_model, train_ds)

# fit using the best model
To1_best_model.fit(train_ds)

# distributons of classes
distributions_of_classes(y_train, y_test)

# actual and predicted y values/ confusion matrix
generate_confusion_matrix(To1_best_model, "To1_best_model", train_ds, test_ds)



