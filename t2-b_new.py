# libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import keras_tuner
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# import autokeras as ak
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import keras_nlp
from tensorflow.keras.optimizers import Adam

# -------------------------------
# version check
# -------------------------------
# from tensorflow import keras

print('tensorflow', tf.__version__)
print('keras', keras.__version__)
print('keras_tuner', keras_tuner.__version__)
print('keras_nlp', keras_nlp.__version__)

# -------------------------------
# read and modify data
# -------------------------------
# read the data
df = pd.read_csv('text-FinancialSentimentAnalysis.csv', encoding='utf-8')
df_original = df.copy()

# one hot encoding
dummies = pd.get_dummies(df['Sentiment'])
df.drop('Sentiment', axis=1, inplace=True)
df = pd.concat([df, dummies], axis=1)

# split into training and test datasets
num = df.shape[0] * 0.7
num = round(num)
df_train = df.loc[:num]
df_test = df.loc[num+1:]

# convert df to tf.dataset with batch 32
target = ['negative', 'neutral', 'positive']

raw_train_ds = tf.data.Dataset.from_tensor_slices((df_train['Sentence'], df_train[target])).batch(batch_size=32)
raw_test_ds = tf.data.Dataset.from_tensor_slices((df_test['Sentence'], df_test[target])).batch(batch_size=32)

# do async prefetching
raw_train_ds = raw_train_ds.cache().prefetch(buffer_size=10)
raw_test_ds = raw_test_ds.cache().prefetch(buffer_size=10)

# ---------------------------------------
# t2-b
# ---------------------------------------
# BertClassifier model
def t2_b_create_model(hp):
  model = keras_nlp.models.BertClassifier.from_preset(
    "bert_base_en_uncased"
    , num_classes=3
    , activation='softmax'
    , dropout=hp.Choice("dropout", values=[0.0,0.2,0.5])
)
  model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=hp.Choice("learning_rate", values=[0.01, 0.001, 0.0001, 0.00001])), metrics=['accuracy'])
  return model

# grid search 
t2_b_model = keras_tuner.GridSearch(
  t2_b_create_model
  , objective='accuracy'
  , overwrite=True
)

# search
num_epochs = 100
t2_b_model.search(raw_train_ds, epochs=num_epochs)

# best model results
print("---------------- best params --------------------")
t2_b_best_param = t2_b_model.get_best_hyperparameters(num_trials=1)[0]
print("learning_rate: ", t2_b_best_param.get("learning_rate"))
print("dropout: ", t2_b_best_param.get("dropout"))

print("---------------- best model results --------------------")
t2_b_best_model = t2_b_model.get_best_models()[0]
t2_b_best_model.build(raw_train_ds)
print(t2_b_best_model.summary())

# fit using the best model
t2_b_best_model.fit(raw_train_ds)

# predict/ actual and predict values
y_actual = []
y_predict = []
for text, label in raw_test_ds:
  y_actual.append(label.numpy())
  y_predict.append(t2_b_best_model.predict(text).argmax(axis=-1))
  
y_actual = np.concatenate(y_actual, axis=0)
y_actual = np.argmax(np.array(y_actual), axis=1)
y_predict = np.concatenate(y_predict, axis=0)

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




