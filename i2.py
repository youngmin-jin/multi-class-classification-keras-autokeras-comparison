# libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import keras_tuner
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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
def i2_create_model(hp):
  inputs = keras.Input(shape=image_size+(3,))       
  
  x = layers.Conv2D(hp.Choice('neuron', values=[100,500,1000,1500,2000]), 3, strides=2, padding='same', activation=hp.Choice('activation', values=['relu','sigmoid','tanh']))(inputs)
  for i in range(hp.Int('hidden_layers', 0, 2)):
    x = layers.Conv2D(hp.Choice('neuron', values=[100,500,1000,1500,2000])
        , 3, strides=2, padding='same'
        , activation=hp.Choice('activation', values=['relu','sigmoid','tanh']))(x)
  x = layers.SeparableConv2D(hp.Choice('neuron', values=[100,500,1000,1500,2000]), 3, padding="same", activation=hp.Choice('activation', values=['relu','sigmoid','tanh']))(x)
  x = layers.GlobalMaxPooling2D()(x)
  x = layers.Dropout(hp.Choice('dropout', values=[0.0,0.2,0.5]))(x)
   
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
num_epochs = 100
i2_model.search(train_ds, epochs=num_epochs)

# best model results
print("---------------- best params --------------------")
i2_best_param = i2_model.get_best_hyperparameters(num_trials=1)[0]
print("neuron: ", i2_best_param.get("neuron"))
print("activation: ", i2_best_param.get("activation"))
print("hidden_layers: ", i2_best_param.get("hidden_layers"))
print("dropout: ", i2_best_param.get("dropout"))

print("---------------- best model results --------------------")
i2_best_model = i2_model.get_best_models(1)[0]
i2_best_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(i2_best_model.summary())

# fit using the best model
i2_best_model.fit(train_ds)

# evaluate on the test dataset
print('----------- Evaluation on Test Dataset ---------------')
test_loss, test_accuracy = i2_best_model.evaluate(test_ds)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# predict/ actual and predict values
y_actual = []
y_predict = []
for img, label in test_ds:
  y_actual.append(label.numpy())
  y_predict.append(i2_best_model.predict(img).argmax(axis=-1))
  
y_actual = np.concatenate(y_actual, axis=0)
y_actual = np.argmax(np.array(y_actual), axis=1)
y_predict = np.concatenate(y_predict, axis=0)

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



