# libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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

x = layers.Embedding(max_features, embedding_dim)(inputs)
x = layers.Conv1D(128, 3, strides=2, padding='same', activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dropout(0.5)(x)

# output layer
predictions = layers.Dense(3, activation='softmax', name='predictions')(x)

# model
t1_model = keras.Model(inputs, predictions)

# compile
t1_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit
num_epochs = 100
t1_model.fit(ks_train_ds, epochs = num_epochs)

# summary
print('----------- results ---------------')
print(t1_model.summary())

# evaluate on the test dataset
print('----------- Evaluation on Test Dataset ---------------')
test_loss, test_accuracy = t1_model.evaluate(ks_test_ds)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# predict/ actual and predict values
y_actual = []
y_predict = []
for text, label in ks_test_ds:
  y_actual.append(label.numpy())
  y_predict.append(t1_model.predict(text).argmax(axis=-1))
  
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

