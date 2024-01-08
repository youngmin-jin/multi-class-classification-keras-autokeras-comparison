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
y_train, y_test, train_ds, test_ds = create_text_input('text-FinancialSentimentAnalysis.csv', False, True)

# -------------------------------
# Tm 
# -------------------------------
# input layer
inputs = keras.Input(shape=(None,), dtype='int64')

# hidden layers
x = keras.layers.Embedding(max_features, embedding_dim)(inputs)
x = keras.layers.Conv1D(128, 3, strides=2, padding='same', activation='relu')(x)
x = keras.layers.GlobalMaxPooling1D()(x)
x = keras.layers.Dropout(0.5)(x)

# output layer
predictions = keras.layers.Dense(3, activation='softmax', name='predictions')(x)

# model
Tm_model = keras.Model(inputs, predictions)

# compile
Tm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit
num_epochs = 3
Tm_model.fit(train_ds, epochs=num_epochs)

# summary
print(Tm_model.summary())

# distributions of classes
distributions_of_classes(y_train, y_test)

# actual and predicted y values/ confusion matrix
generate_confusion_matrix(Tm_model, "Tm_model", train_ds, test_ds)

