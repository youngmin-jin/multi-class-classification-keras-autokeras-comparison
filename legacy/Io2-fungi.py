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
    x = keras.layers.RandomFlip('horizontal')(inputs)
    x = keras.layers.RandomRotation(0.005)(x)
    x = keras.layers.RandomContrast(hp.Choice('random_contrast', values=[0.002,0.01]))(x)
    x = keras.layers.BatchNormalization()(x) 
    x = keras.layers.Resizing(224,224)(x)   
    x = keras.applications.EfficientNetB7(
        input_shape=(224,224,3)
        , include_top=False
        , weights='imagenet'
        , drop_connect_rate=0.005
        , pooling='avg'
    )(x)  
    x = keras.layers.Dropout(0.5)(x)
  
    # output layer
    outputs = keras.layers.Dense(5, activation='softmax')(x)  
  
    model = keras.Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=3e-5), metrics=['accuracy'])
    return model
    
  def fit(self, hp, model, *args, **kwargs):
    return model.fit(*args, batch_size=64, **kwargs)


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
num_epochs = 100
Io2_model.search(x_train, y_train, epochs=num_epochs, validation_split=0.2, callbacks=[es])
# Io2_model.search(x_train, y_train, epochs=num_epochs)

# best parameters
get_best_params(Io2_model, "random_contrast")

# best model summary 
Io2_best_model = get_best_model(Io2_model, x_train, y_train)

# fit using the best model
Io2_best_model.fit(x_train, y_train)

# distributions of classes
distributions_of_classes(y_train, y_test)

# actual and predicted y values/ confusion matrix
generate_confusion_matrix(Io2_best_model, "Io2_best_model", x_test, y_test)



