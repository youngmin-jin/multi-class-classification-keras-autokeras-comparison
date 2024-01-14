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
x_train, x_test, y_train, y_test = create_image_input('Multi-class Weather Dataset', True)

# -------------------------------
# Im
# -------------------------------
# input layer
inputs = keras.Input(shape=image_size+(3,))

# hidden layers
x = keras.layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(inputs)
x = keras.layers.SeparableConv2D(128, 3, padding="same", activation='relu')(x)
x = keras.layers.GlobalMaxPooling2D()(x)
x = keras.layers.Dropout(0.5)(x)

# output layer
outputs = keras.layers.Dense(4, activation='softmax')(x)

# model
Im_model = keras.Model(inputs, outputs)

# compile
Im_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit
num_epochs = 3
Im_model.fit(x_train, y_train, epochs=num_epochs)

# summary
print(Im_model.summary())

# distributions of classes
distributions_of_classes(y_train, y_test)

# actual and predicted y values/ confusion matrix
generate_confusion_matrix(Im_model, "Im_model", x_test, y_test)

