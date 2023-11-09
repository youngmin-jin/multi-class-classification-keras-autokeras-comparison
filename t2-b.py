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

# import tensorflow_hub as hub
# import tensorflow_text as text
import keras_nlp
# from tensorflow.keras.models import load_model

# -------------------------------
# version check
# -------------------------------
# from tensorflow import keras

print('tensorflow', tf.__version__)
print('keras', keras.__version__)
# print('keras_tuner', keras_tuner.__version__)
# print('tensorflow_hub', hub.__version__)
print('keras_nlp', keras_nlp.__version__)

# -------------------------------
# read and modify data
# -------------------------------
# read the data
df = pd.read_csv('text-test.csv', encoding='utf-8')
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

# Split into features and target
target = ['negative', 'neutral', 'positive']
# x = df['Sentence']
# y = df[target]

# Split into training and test sets
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)

raw_train_ds = tf.data.Dataset.from_tensor_slices((df_train['Sentence'], df_train[target])).batch(batch_size=32)
raw_test_ds = tf.data.Dataset.from_tensor_slices((df_test['Sentence'], df_test[target])).batch(batch_size=32)

# prepare the setting
# max_features = 20000
# embedding_dim = 128
# sequency_length = 500

# tokenize
# def tokenize_text(text, label):
#   tokenizer = keras_nlp.models.BertTokenizer.from_preset("bert_base_en_uncased")
#   return tokenizer(text), label

# map the tokenied text to dict input format
# def map_to_bert_input(tokenized_text, label):
#   input_dict = {
#     'input_word_ids': tokenized_text,
#     'input_mask': tf.ones_like(tokenized_text),
#     'input_type_ids': tf.zeros_like(tokenized_text)   
#   }
#   return input_dict, label

# apply them to x_train
# ks_train_ds = raw_train_ds.map(tokenize_text)
# ks_train_ds = ks_train_ds.map(map_to_bert_input)

# ks_test_ds = raw_test_ds.map(tokenize_text)
# ks_test_ds = ks_test_ds.map(map_to_bert_input)

# ---------------------------------------
# t2-b
# ---------------------------------------
# BertClassifier
t2_b_model = keras_nlp.models.BertClassifier.from_preset(
  "bert_base_en_uncased"
  , num_classes=3
)

# compile
t2_b_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit
num_epochs = 5
t2_b_model.fit(raw_train_ds, epochs=num_epochs)

# summary
print('----------- results ---------------')
print(t2_b_model.summary())


# actual and predict values
# print("----------------- y_actual ----------------------")
# y_actual = np.argmax(np.array(y_test), axis=1)
# print(y_actual)

# print("----------------- y_predict ----------------------")
# y_predict = t2_b_model.predict(x_test)
# y_predict = y_predict.argmax(axis=-1)
# print(y_predict)

# predict/ actual and predict values
y_actual = []
y_predict = []
for text, label in raw_test_ds:
  y_actual.append(label.numpy())
  y_predict.append(t2_b_model.predict(text).argmax(axis=-1))
  
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




