# libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from input_output import *

# -------------------------------
# version check
# -------------------------------
print('tensorflow', tf.__version__)
print('keras', keras.__version__)

# -------------------------------
# read the data
# -------------------------------
x_train, x_test, y_train, y_test = create_structured_input("structured-bodyPerformance.csv", True)

# -------------------------------
# Sm 
# -------------------------------
# model
Sm_model = keras.models.Sequential()

# layers
Sm_model.add(keras.layers.Dense(128, input_shape=(x_train.shape[1],), activation='relu'))
Sm_model.add(keras.layers.Dense(128, activation='relu'))
Sm_model.add(keras.layers.Dense(128, activation='relu'))
Sm_model.add(keras.layers.Dropout(0.5))
Sm_model.add(keras.layers.Dense(4, activation='softmax'))
  
# compile
Sm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit
num_epochs = 100
Sm_model.fit(x_train, y_train, epochs=num_epochs)

# summary 
print(Sm_model.summary())

# distributions of classes
distributions_of_classes(y_train, y_test)

# actual and predicted y values/ confusion matrix
generate_confusion_matrix(Sm_model, "Sm_model", x_test, y_test)
