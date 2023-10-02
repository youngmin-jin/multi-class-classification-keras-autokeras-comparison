# libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from kerastuner.tuners import RandomSearch

# -------------------------------
# read and modify data
# -------------------------------
# generate a dataset
image_size = (180, 180)
batch_size = 128
train_ds = tf.keras.utils.image_dataset_from_directory(
    'Multi-class Weather Dataset'
    , validation_split=0.3
    , subset='training'
    , seed=1337
    , image_size=image_size
    , batch_size=batch_size
    , label_mode="categorical"
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    'Multi-class Weather Dataset'
    , validation_split=0.3
    , subset='validation'
    , seed=1337
    , image_size=image_size
    , batch_size=batch_size
    , label_mode="categorical"
)

# data augmentation model
data_augmentation = keras.Sequential(
    [keras.layers.RandomFlip('horizontal')
      , keras.layers.RandomRotation(0.1)
    ]
)

# apply data augmentation to the training dataset
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label)
    , num_parallel_calls = tf.data.AUTOTUNE
)
    
# prefetch samples in GPU memeory and maximize GPU utilization
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# -------------------------------
# i2 
# -------------------------------
print("------------------ model------------------")

# create a model
def i2_create_model(hp):
  inputs = keras.Input(shape=image_size+(3,))       
  
  # ------------------------ legacy -----------------------------------------------
  # x = layers.Rescaling(1.0/255)(inputs)                         # rescaling the RGB channels
  # x = layers.Conv2D(hp.Choice('neurons', values=[100,500,1000]), 3, strides=2, padding='same')(x)
  # x = layers.BatchNormalization()(x)
  # x = layers.Activation(hp.Choice('activation', values=['relu','sigmoid','tanh']))(x)
                    
  # previous_block_activation = x
  
  # x = layers.Activation(hp.Choice('activation', values=['relu','sigmoid','tanh']))(x)
  # x = layers.SeparableConv2D(hp.Choice('neurons', values=[100,500,1000]), 3, padding="same")(x)
  # x = layers.BatchNormalization()(x)

  # x = layers.Activation(hp.Choice('activation', values=['relu','sigmoid','tanh']))(x)
  # x = layers.SeparableConv2D(hp.Choice('neurons', values=[100,500,1000]), 3, padding="same")(x)
  # x = layers.BatchNormalization()(x)

  # x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

  # Project residual
  # residual = keras.layers.Conv2D(hp.Choice('neurons', values=[100,500,1000]), 1, strides=2, padding="same")(
  #   previous_block_activation
  # )
  # x = keras.layers.add([x, residual])  # Add back residual
  # previous_block_activation = x  # Set aside next residual

  # x = keras.layers.GlobalAveragePooling2D()(x)
  # x = keras.layers.Dropout(hp.Choice('dropout',values=[0.0,0.2,0.5]))(x)      
  # -------------------------------------------------------------------------------
  
  x = layers.Conv2D(hp.Choice('neuron', values=[100,500,1000]), 3, strides=2, padding='same', activation=hp.Choice('activation', values=['relu','sigmoid','tanh']))(inputs)
  for i in range(hp.Int('hidden_layers', 0, 2)):
    x = layers.Conv2D(hp.Choice('neuron', values=[100,500,1000])
        , 3, strides=2, padding='same'
        , activation=hp.Choice('activation', values=['relu','sigmoid','tanh']))(x)
  x = layers.SeparableConv2D(hp.Choice('neuron', values=[100,500,1000]), 3, padding="same", activation=hp.Choice('activation', values=['relu','sigmoid','tanh']))(x)
  x = layers.GlobalMaxPooling2D()(x)
  x = layers.Dropout(hp.Choice('dropout', values=[0.0,0.2,0.5]))(x)
   
  outputs = keras.layers.Dense(4, activation='softmax')(x)  
  model = keras.Model(inputs, outputs)
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',keras.metrics.Precision(), keras.metrics.Recall()])
  return model

# random search
i2_model = RandomSearch(
  i2_create_model
  , objective='val_accuracy'
  , max_trials=1000
  , overwrite=True
  , directory='i2_tuner1'
  , project_name='i2_tuner1'
)

# fit
num_epochs = 300
i2_model.search(train_ds, validation_data=test_ds, epochs=num_epochs)

# summary 
print("---------------- results --------------------")
print(i2_model.results_summary())
print(i2_model.get_best_models()[0])





