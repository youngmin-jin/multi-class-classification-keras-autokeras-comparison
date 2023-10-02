# libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

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

# --------------------- legacy -------------------------------------
# x = layers.Rescaling(1.0/255)(inputs)                         # rescaling the RGB channels
# x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
# x = layers.BatchNormalization()(x)
# x = layers.Activation("relu")(x)

# previous_block_activation = x

# x = layers.Activation("relu")(x)
# x = layers.SeparableConv2D(128, 3, padding="same")(x)
# x = layers.BatchNormalization()(x)

# x = layers.Activation("relu")(x)
# x = layers.SeparableConv2D(128, 3, padding="same")(x)
# x = layers.BatchNormalization()(x)

# x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

# # Project residual
# residual = layers.Conv2D(128, 1, strides=2, padding="same")(
#     previous_block_activation
# )
# x = layers.add([x, residual])  # Add back residual
# previous_block_activation = x  # Set aside next residual

# x = layers.SeparableConv2D(128, 3, padding="same")(x)
# x = layers.BatchNormalization()(x)
# x = layers.Activation("relu")(x)

# x = layers.GlobalAveragePooling2D()(x)
# x = layers.Dropout(0.5)(x)
# -------------------------------------------------------------------

x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(inputs)
x = layers.SeparableConv2D(128, 3, padding="same", activation='relu')(x)
x = layers.GlobalMaxPooling2D()(x)
x = layers.Dropout(0.5)(x)

# output layer
outputs = layers.Dense(4, activation='softmax')(x)

# model
i1_model = keras.Model(inputs, outputs)

# compile
i1_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',keras.metrics.Precision(), keras.metrics.Recall()])

# fit
num_epochs = 300
i1_history = i1_model.fit(train_ds, validation_data=test_ds, epochs=num_epochs)

# summary
print("---------------- results --------------------")
print(i1_model.summary())

# best score
print('i1 best loss:', max(i1_history.history['loss']))
print('i1 best val_loss:', max(i1_history.history['val_loss']))
print('i1 best accuracy:', max(i1_history.history['accuracy']))
print('i1 best val_accuracy:', max(i1_history.history['val_accuracy']))
print('i1 best precision:', max(i1_history.history['precision']))
print('i1 best val_precision:', max(i1_history.history['val_precision']))
print('i1 best recall:', max(i1_history.history['recall']))
print('i1 best val_recall:', max(i1_history.history['val_recall']))
