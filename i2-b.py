# libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner
from input_output import *

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
x_train, x_test, y_train, y_test = create_image_input("i2-b")

# -------------------------------
# i2 
# -------------------------------
def i2_create_model(hp):
  image_size = (180, 180)
  inputs = keras.Input(shape=image_size+(3,))   
  
  x = keras.applications.Xception(include_top=False, weights="imagenet")(inputs)
  for i in range(hp.Int('hidden_layers', 0, 1)):
    x = layers.GlobalAveragePooling2D()(x)
  
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
num_epochs = 3
i2_model.search(x_train, y_train, epochs=num_epochs)

# best model results
print("---------------- best params --------------------")
i2_best_param = i2_model.get_best_hyperparameters(num_trials=1)[0]
# print("neuron: ", i2_best_param.get("neuron"))
# print("activation: ", i2_best_param.get("activation"))
print("hidden_layers: ", i2_best_param.get("hidden_layers"))
# print("dropout: ", i2_best_param.get("dropout"))

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

print("----------------- y_actual ----------------------")
print(y_actual)

print("----------------- y_predict ----------------------")
print(y_predict)

# confusion matrix
print("----------------- confusion matrix ----------------------")
print(confusion_matrix(y_actual, y_predict))

# confusion report
print("----------------- confusion report ----------------------")
print(classification_report(y_actual, y_predict))



