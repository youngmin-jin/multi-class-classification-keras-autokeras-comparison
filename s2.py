# libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import keras_tuner
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
# s2 
# -------------------------------
# build a sequential model
def s2_create_model(hp):
  model = keras.models.Sequential()
  model.add(keras.layers.Dense(hp.Choice('neuron', values=[100,500,1000])
  , input_shape=(df[predictors].shape[1],)
  , activation=hp.Choice('activation', values=['relu','sigmoid','tanh'])))
  for i in range(hp.Int('hidden_layers', 2, 4)):
    model.add(keras.layers.Dense(hp.Choice('neuron', values=[100,500,1000])
    , activation=hp.Choice('activation', values=['relu','sigmoid','tanh'])))
  model.add(keras.layers.Dropout(hp.Choice('dropout', values=[0.0,0.2,0.5])))
  model.add(keras.layers.Dense(4, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',keras.metrics.Precision(), keras.metrics.Recall()])
  return model

# make a model
s2_model = keras_tuner.GridSearch(
  s2_create_model
  , objective='val_accuracy'
  , overwrite=True
)

# search
num_epochs = 300
s2_model.search(ks_x_train, ks_y_train, validation_data=(ks_x_test, ks_y_test), epochs=num_epochs)

# best model results
print("---------------- best params --------------------")
s2_best_param = s2_model.get_best_hyperparameters(num_trials=1)[0]
print("neuron: ", s2_best_param.get("neuron"))
print("activation: ", s2_best_param.get("activation"))
print("hidden_layers: ", s2_best_param.get("hidden_layers"))
print("dropout: ", s2_best_param.get("dropout"))

print("---------------- best model results --------------------")
s2_best_model = s2_model.get_best_models()[0]
s2_best_model.build(ks_x_train.shape)
print(s2_best_model.summary())

# fit using the best model
s2_best_model.fit(ks_x_train, ks_y_train, epochs=num_epochs)

# actual and predicted value
print("----------------- y_actual ----------------------")
y_actual = np.argmax(np.array(ks_y_test), axis=1)
print(y_actual)

print("----------------- y_predict ----------------------")
y_predict = s2_best_model.predict(ks_x_test)
y_predict = y_predict.argmax(axis=-1)
print(y_predict)

# confusion matrix
print("----------------- confusion matrix ----------------------")
print(confusion_matrix(y_actual, y_predict))

# confusion report
print("----------------- confusion report ----------------------")
print(classification_report(y_actual, y_predict))











