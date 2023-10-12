# libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
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
    [layers.RandomFlip('horizontal')
      , layers.RandomRotation(0.1)
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
# i1
# -------------------------------
inputs = keras.Input(shape=image_size+(3,))

x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(inputs)
x = layers.SeparableConv2D(128, 3, padding="same", activation='relu')(x)
x = layers.GlobalMaxPooling2D()(x)
x = layers.Dropout(0.5)(x)

# output layer
outputs = layers.Dense(4, activation='softmax')(x)

# model
i1_model = keras.Model(inputs, outputs)

# compile
i1_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# fit
num_epochs = 100
i1_model.fit(train_ds, epochs=num_epochs)

# summary
print('----------- results ---------------')
print(i1_model.summary())

# evaluate on the test dataset
print('----------- Evaluation on Test Dataset ---------------')
test_loss, test_accuracy = i1_model.evaluate(test_ds)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# predict/ actual and predict values
y_actual = []
y_predict = []
for img, label in test_ds:
  y_actual.append(label.numpy())
  y_predict.append(i1_model.predict(img).argmax(axis=-1))
  
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


