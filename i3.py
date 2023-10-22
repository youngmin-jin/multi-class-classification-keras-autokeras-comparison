# libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import autokeras as ak
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# -------------------------------
# read and modify data
# -------------------------------
# generate a dataset
ak_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'image-weather'
    , validation_split=0.3
    , subset='training'
    , seed=1337
    , label_mode="categorical"
)
ak_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'image-weather'
    , validation_split=0.3
    , subset='validation'
    , seed=1337
    , label_mode="categorical"
)

# data augmentation model
data_augmentation = keras.Sequential(
    [tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal')
      , tf.keras.layers.experimental.preprocessing.RandomRotation(0.1)
    ]
)

# apply data augmentation to the training dataset
ak_train_ds = ak_train_ds.map(
    lambda img, label: (data_augmentation(img), label)
    , num_parallel_calls = tf.data.AUTOTUNE
)

# -------------------------------
# i3
# -------------------------------
# model
print("-------------- model ------------------")
num_epochs = 100
i3_model = ak.ImageClassifier(overwrite=True, num_classes=4, metrics=['accuracy'])
i3_model.fit(ak_train_ds, epochs=num_epochs)

# evaluate on the test dataset
print('----------- Evaluation on Test Dataset ---------------')
test_loss, test_accuracy = i3_model.evaluate(ak_test_ds)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# predict/ actual and predict values
y_actual = []
for img, label in ak_test_ds:
  y_actual.append(label.numpy())  
y_actual = np.concatenate(y_actual, axis=0)
y_actual = np.argmax(np.array(y_actual), axis=1)

y_predict = i3_model.predict(ak_test_ds).argmax(axis=-1)
y_predict_train = i3_model.predict(ak_train_ds).argmax(axis=-1)
# y_predict = np.concatenate(y_predict, axis=0)

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

# summary 
print("---------------- results --------------------")
i3_model_result = i3_model.export_model()
print(i3_model_result.summary())
