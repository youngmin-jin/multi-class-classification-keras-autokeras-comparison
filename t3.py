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
df = pd.read_csv('text-FinancialSentimentAnalysis.csv', encoding='utf-8')

# -------------------------------
# t3
# -------------------------------
# data
ak_x_train, ak_x_test, ak_y_train, ak_y_test = train_test_split(df[['Sentence']], df[['Sentiment']], test_size= 0.3)

ak_x_train = np.array(ak_x_train)
ak_y_train = np.array(ak_y_train)
ak_x_test = np.array(ak_x_test)
ak_y_test = np.array(ak_y_test)

# model
num_epochs = 100
t3_model = ak.TextClassifier(overwrite=True, num_classes=3, metrics=['accuracy'])
t3_model.fit(ak_x_train, ak_y_train, epochs=num_epochs)

# actual and predicted value
print("----------------- y_actual ----------------------")
y_actual = ak_y_test
print(y_actual)

print("----------------- y_predict ----------------------")
y_predict = t3_model.predict(ak_x_test)
print(y_predict)

# confusion matrix
print("----------------- confusion matrix ----------------------")
print(confusion_matrix(y_actual, y_predict))

# confusion report
print("----------------- confusion report ----------------------")
print(classification_report(y_actual, y_predict))

# summary 
print("---------------- results --------------------")
t3_model_result = t3_model.export_model()
print(t3_model_result.summary())
