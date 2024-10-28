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
class Io2_create_model(keras_tuner.HyperModel):
  def build(self, hp):  
    # input layer
    inputs = keras.Input(shape=image_size+(3,)) 

    # hidden layers
    x = keras.layers.RandomFlip(hp.Choice("random_flip", values=['horizontal','horizontal_and_vertical']))(inputs)
    x = keras.layers.RandomRotation(hp.Choice("random_rotation", values=[0.005,0.007,0.01]))(x)
    x = keras.layers.RandomContrast(hp.Choice('random_contrast', values=[0.002,0.003,0.007]))(x)
    x = keras.layers.BatchNormalization()(x) 
    x = keras.layers.Resizing(224,224)(x)   
    x = keras.applications.EfficientNetB7(
        input_shape=(224,224,3)
        , include_top=False
        , weights='imagenet'
        , drop_connect_rate=hp.Choice("drop_connect_rate", values=[0.005,0.01,0.1])
        , pooling='avg'
    )(x)  
    x = keras.layers.Dropout(hp.Choice("dropout", values=[0.0,0.2,0.5]))(x)
  
    # output layer
    outputs = keras.layers.Dense(5, activation='softmax')(x)  
  
    model = keras.Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=hp.Choice("learning_rate", values=[3e-4,1e-4,5e-5,3e-5,1e-5])), metrics=['accuracy'])
    return model
    
  def fit(self, hp, model, *args, **kwargs):
    return model.fit(*args, batch_size=hp.Choice("batch_size", values=[16,64]), **kwargs)


# apply grid search
Io2_model = keras_tuner.GridSearch(
  Io2_create_model()
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
get_best_params(Io2_model, "random_flip","random_rotation","random_contrast","drop_connect_rate","dropout","batch_size")

# best model summary 
Io2_best_model = get_best_model(Io2_model, x_train, y_train)

# fit using the best model
Io2_best_model.fit(x_train, y_train)

# distributions of classes
distributions_of_classes(y_train, y_test)

# actual and predicted y values/ confusion matrix
generate_confusion_matrix(Io2_best_model, "Io2_best_model", x_test, y_test)



