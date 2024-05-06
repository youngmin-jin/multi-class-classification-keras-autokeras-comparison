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
x_train, x_test, y_train, y_test = create_image_input('fungi',True)

# -------------------------------
# Io2
# -------------------------------
# def Io2_create_model(hp):
#   # image_size = (224,224)
  
#   # input layer
#   inputs = keras.Input(shape=image_size+(3,)) 

#   # hidden layers
#   x = keras.layers.RandomFlip(hp.Choice("random_flip", values=['vertical','horizontal','horizontal_and_vertical']))(inputs)
#   x = keras.layers.RandomRotation(hp.Choice("random_rotation", values=[0.003,0.005,0.01,0.03]))(x)
#   x = keras.layers.BatchNormalization(momentum=hp.Choice("momentum", values=[0.5,0.99]))(x)
#   # x = keras.layers.BatchNormalization()(x)
#   x = keras.layers.Resizing(224,224)(x)
#   x = keras.applications.EfficientNetB7(
#       input_shape=(224,224,3)
#       , include_top=False
#       , weights='imagenet'
#       , pooling=hp.Choice("pooling", values=['avg', 'max'])
#       , drop_connect_rate=hp.Choice("drop_connect_rate", values=[0.1,0.5])
#   )(x)

#   # output layer
#   outputs = keras.layers.Dense(5, activation='softmax')(x)  

#   model = keras.Model(inputs, outputs)
#   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#   return model

def Io2_create_model(hp):
  inputs = keras.Input(shape=image_size+(3,)) 
  model = keras.applications.EfficientNetB7(include_top=False, input_tensor=inputs, weights="imagenet")
  
  model.trainable = False

  x = keras.layers.GlobalAveragePooling2D()(model.output)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Dropout(hp.Choice("dropout", values=[0.2,0.5]))(x)
  outputs = keras.layers.Dense(5, activation="softmax")(x)
  
  model = keras.Model(inputs, outputs, name="EfficientNet")
  model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=hp.Choice("learning_rate", values=[3e-4, 1e-4, 5e-5, 3e-5])), metrics=["accuracy"])
  return model

# apply grid search
Io2_model = keras_tuner.GridSearch(
  Io2_create_model
  , objective='accuracy'
  , overwrite=True
)

# early stopping
es = keras.callbacks.EarlyStopping(
  monitor="val_accuracy"
  , patience=5
  , restore_best_weights=True
)

# search
num_epochs = 50
Io2_model.search(x_train, y_train, epochs=num_epochs, validation_split=0.2, callbacks=[es])
# Io2_model.search(x_train, y_train, epochs=num_epochs)

# best parameters
get_best_params(Io2_model, "dropout", "learning_rate")

# best model summary 
Io2_best_model = get_best_model(Io2_model, x_train, y_train)

# fit using the best model
Io2_best_model.fit(x_train, y_train)

# distributions of classes
distributions_of_classes(y_train, y_test)

# actual and predicted y values/ confusion matrix
generate_confusion_matrix(Io2_best_model, "Io2_best_model", x_test, y_test)



