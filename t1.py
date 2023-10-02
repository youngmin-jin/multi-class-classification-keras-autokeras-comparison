# libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

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
# t1 
# -------------------------------
# a integer input for vocab indices
inputs = tf.keras.Input(shape=(None,), dtype='int64')

# ------------- legacy --------------------------------------
# add a layer to map those vocab indices into a space of dimensionality
# x = keras.layers.Embedding(max_features, embedding_dim)(inputs)
# x = keras.layers.Dropout(0.5)(x)

# Conv1D + global max pooling
# x = keras.layers.Conv1D(128, 7, padding='valid', activation='relu', strides=3)(x)
# x = keras.layers.Conv1D(128, 7, padding='valid', activation='relu', strides=3)(x)
# x = keras.layers.GlobalMaxPooling1D()(x)

# fadd a vanilla hidden layer
# x = keras.layers.Dense(128, activation='relu')(x)
# x = keras.layers.Dropout(0.5)(x)
# -----------------------------------------------------------

x = layers.Embedding(max_features, embedding_dim)(inputs)
x = layers.Conv1D(128, 3, strides=2, padding='same', activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dropout(0.5)(x)

# output layer
predictions = layers.Dense(3, activation='softmax', name='predictions')(x)

# model
t1_model = keras.Model(inputs, predictions)

# compile
t1_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',keras.metrics.Precision(), keras.metrics.Recall()])

# fit
num_epochs = 300
t1_history = t1_model.fit(ks_train_ds, validation_data=ks_test_ds, epochs = num_epochs)

# summary
print('----------- results ---------------')
print(t1_model.summary())

# best score
print('t1 best loss:', max(t1_history.history['loss']))
print('t1 best val_loss:', max(t1_history.history['val_loss']))
print('t1 best accuracy:', max(t1_history.history['accuracy']))
print('t1 best val_accuracy:', max(t1_history.history['val_accuracy']))
print('t1 best precision:', max(t1_history.history['precision']))
print('t1 best val_precision:', max(t1_history.history['val_precision']))
print('t1 best recall:', max(t1_history.history['recall']))
print('t1 best val_recall:', max(t1_history.history['val_recall']))
