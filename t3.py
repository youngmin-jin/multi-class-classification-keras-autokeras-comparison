# libraries 
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import autokeras as ak

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
num_epochs = 300
t3_model = ak.TextClassifier(overwrite=True, max_trials=1, num_classes=3, metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
t3_history = t3_model.fit(ak_x_train, ak_y_train, validation_data=(ak_x_test, ak_y_test), epochs=num_epochs)

# summary 
print("---------------- results --------------------")
t3_model_result = t3_model.export_model()
print(t3_model_result.summary())

# best score
print('t3 best loss:', max(t3_history.history['loss']))
print('t3 best val_loss:', max(t3_history.history['val_loss']))
print('t3 best accuracy:', max(t3_history.history['accuracy']))
print('t3 best val_accuracy:', max(t3_history.history['val_accuracy']))
print('t3 best precision:', max(t3_history.history['precision']))
print('t3 best val_precision:', max(t3_history.history['val_precision']))
print('t3 best recall:', max(t3_history.history['recall']))
print('t3 best val_recall:', max(t3_history.history['val_recall']))
