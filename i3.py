# libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import autokeras as ak

# -------------------------------
# read and modify data
# -------------------------------
# generate a dataset
ak_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'Multi-class Weather Dataset'
    , validation_split=0.3
    , subset='training'
    , seed=1337
    , label_mode="categorical"
)
ak_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'Multi-class Weather Dataset'
    , validation_split=0.3
    , subset='validation'
    , seed=1337
    , label_mode="categorical"
)

# -------------------------------
# i3
# -------------------------------
# model
print("-------------- model ------------------")
num_epochs = 300
i3_model = ak.ImageClassifier(overwrite=True, max_trials=1, num_classes=4, metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
i3_history = i3_model.fit(ak_train_ds, validation_data=ak_test_ds, epochs=num_epochs)

# summary 
print("---------------- results --------------------")
i3_model_result = i3_model.export_model()
print(i3_model_result.summary())

# best score
print('i3 best loss:', max(i3_history.history['loss']))
print('i3 best val_loss:', max(i3_history.history['val_loss']))
print('i3 best accuracy:', max(i3_history.history['accuracy']))
print('i3 best val_accuracy:', max(i3_history.history['val_accuracy']))
print('i3 best precision:', max(i3_history.history['precision']))
print('i3 best val_precision:', max(i3_history.history['val_precision']))
print('i3 best recall:', max(i3_history.history['recall']))
print('i3 best val_recall:', max(i3_history.history['val_recall']))
