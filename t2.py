# libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from kerastuner.tuners import RandomSearch

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
target = ['negative','neutral','positive']

raw_train_ds = tf.data.Dataset.from_tensor_slices((df_train['Sentence'], df_train[target])).batch(batch_size=32)
raw_test_ds = tf.data.Dataset.from_tensor_slices((df_test['Sentence'], df_test[target])).batch(batch_size=32)

# prepare the setting
max_features = 20000
embedding_dim = 128
sequency_length = 500

vectorize_layer = tf.keras.layers.TextVectorization(
    standardize='lower_and_strip_punctuation'                   # lower the case and remove punctuation
    , max_tokens=max_features
    , output_mode='int'
    , output_sequence_length=sequency_length
)

for text, label in raw_train_ds.take(1):
  for i in range(4):
    print(text.numpy()[i])
    print(label.numpy()[i])

# make a text-only dataset
text_ds = raw_train_ds.map(lambda x,y: x)

# call adapt on a text-only dataset to create the vacabulary
vectorize_layer.adapt(text_ds)

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

# vectorize the data
ks_train_ds = raw_train_ds.map(vectorize_text)
ks_test_ds = raw_test_ds.map(vectorize_text)

# do async prefetching
ks_train_ds = ks_train_ds.cache().prefetch(buffer_size=10)
ks_test_ds = ks_test_ds.cache().prefetch(buffer_size=10)

# -------------------------------
# t2
# -------------------------------
def t2_create_model(hp):
  inputs = tf.keras.Input(shape=(None,), dtype='int64')
  # ---------------------- legacy ---------------------------------------------
  # add a layer to map those vocab indices into a space of dimensionality
  # x = keras.layers.Embedding(max_features, embedding_dim)(inputs)
  # x = keras.layers.Dropout(hp.Choice('dropout', values=[0.0,0.2,0.5]))(x)
  
  # Conv1D + global max pooling
  # x = keras.layers.Conv1D(hp.Choice('neurons', values=[100,500,1000]), 7, padding='valid', activation=hp.Choice('activation', values=['relu','sigmoid','tanh']), strides=3)(x)
  # x = keras.layers.GlobalMaxPooling1D()(x)
  # x = keras.layers.Dense(hp.Choice('neurons', values=[100,500,1000]), activation=hp.Choice('activation', values=['relu','sigmoid','tanh']))(x)
  # x = keras.layers.Dropout(hp.Choice('dropout', values=[0.0,0.2,0.5]))(x)
  # ---------------------------------------------------------------------------
  
  x = keras.layers.Embedding(max_features, embedding_dim)(inputs) 
  for i in range(hp.Int('hidden_layers', 1, 3)):
    x = keras.layers.Conv1D(hp.Choice('neuron', values=[100,500,1000])
    , 3, strides=2, padding='same'
    , activation=hp.Choice('activation', values=['relu','sigmoid','tanh']))(x)
  x = keras.layers.GlobalMaxPooling1D()(x)
  x = keras.layers.Dropout(hp.Choice('dropout', values=[0.0,0.2,0.5]))(x)

  predictions = keras.layers.Dense(3, activation='softmax', name='predictions')(x)
  model = tf.keras.Model(inputs, predictions)
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',keras.metrics.Precision(), keras.metrics.Recall()])
  return model

# randomsearch
t2_model = RandomSearch(
  t2_create_model
  , objective='val_accuracy'
  , max_trials=1000
  , overwrite=True
  , directory='t2_tuner1'
  , project_name='t2_tuner2'
)

# fit
num_epochs = 300
t2_model.search(ks_train_ds, validation_data=(ks_test_ds), epochs=num_epochs)

# summary 
print("---------------- results --------------------")
print(t2_model.results_summary())
print(t2_model.get_best_models()[0])
