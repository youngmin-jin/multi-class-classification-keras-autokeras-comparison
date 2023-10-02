# libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# -------------------------------
# read and modify data
# -------------------------------
# read the data
df = pd.read_csv('structured-bodyPerformance.csv', encoding='utf-8')
df = df.rename({'body fat_%':'body_fat_per', 'sit and bend forward_cm':'sit_and_bend_forward_cm', 'sit-ups counts':'sit_ups_counts', 'broad jump_cm':'broad_jump_cm'}, axis=1)
df_original = df.copy()

# one hot encoding
dummies = pd.get_dummies(df['gender'])
df.drop('gender', axis=1, inplace=True)
df = pd.concat([df, dummies], axis=1)

dummies = pd.get_dummies(df['class'])
df.drop('class', axis=1, inplace=True)
df = pd.concat([df, dummies], axis=1)

# split into training and test datasets
target = ['A','B','C','D']
predictors = np.setdiff1d(df.columns.to_numpy(), target)
ks_x_train, ks_x_test, ks_y_train, ks_y_test = train_test_split(df[predictors], df[target], train_size=0.7)

# -------------------------------
# s1 
# -------------------------------
# build a sequential model
s1_model = keras.models.Sequential()
s1_model.add(keras.layers.Dense(128, input_shape=(df[predictors].shape[1],), activation='relu'))
s1_model.add(keras.layers.Dense(128, activation='relu'))
s1_model.add(keras.layers.Dense(128, activation='relu'))
s1_model.add(keras.layers.Dropout(0.5))
s1_model.add(keras.layers.Dense(4, activation='softmax'))
  
# compile
s1_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit
num_epochs = 300
s1_model.fit(ks_x_train, ks_y_train, epochs=num_epochs)

# actual and predicted value
print("----------------- y_actual ----------------------")
y_actual = np.argmax(np.array(ks_y_test), axis=1)
print(y_actual)

print("----------------- y_predict ----------------------")
y_predict = s1_model.predict(ks_x_test)
y_predict = y_predict.argmax(axis=-1)
print(y_predict)

# confusion matrix
print("----------------- confusion matrix ----------------------")
print(confusion_matrix(y_actual, y_predict))

# confusion report
print("----------------- confusion report ----------------------")
print(classification_report(y_actual, y_predict))



