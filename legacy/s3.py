# libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import autokeras as ak
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# -------------------------------
# version check
# -------------------------------
import tensorflow as tf
from tensorflow import keras
import keras_tuner
import autokeras as ak

print('tensorflow', tf.__version__)
print('keras', keras.__version__)
print('keras_tuner', keras_tuner.__version__)
print('autokeras', ak.__version__)

# -------------------------------
# read and modify data
# -------------------------------
# read the data
df = pd.read_csv('structured-bodyPerformance.csv', encoding='utf-8')
df = df.rename({'body fat_%':'body_fat_per', 'sit and bend forward_cm':'sit_and_bend_forward_cm', 'sit-ups counts':'sit_ups_counts', 'broad jump_cm':'broad_jump_cm'}, axis=1)

# -------------------------------
# s3
# -------------------------------
# data
ak_x_train, ak_x_test, ak_y_train, ak_y_test = train_test_split(df.drop('class', axis=1), df[['class']], test_size= 0.3)

# model
num_epochs = 100
s3_model = ak.StructuredDataClassifier(num_classes=4, metrics=['accuracy'], overwrite=True)
s3_model.fit(ak_x_train, ak_y_train, epochs=num_epochs)

# actual and predicted value
print("----------------- y_actual ----------------------")
y_actual = ak_y_test.to_numpy()
print(y_actual)

print("----------------- y_predict ----------------------")
y_predict = s3_model.predict(ak_x_test)
print(y_predict)

# confusion matrix
print("----------------- confusion matrix ----------------------")
print(confusion_matrix(y_actual, y_predict))

# confusion report
print("----------------- confusion report ----------------------")
print(classification_report(y_actual, y_predict))

# summary 
print("---------------- results --------------------")
s3_model_result = s3_model.export_model()
print(s3_model_result.summary())



















