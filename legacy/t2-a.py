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
  
  x = keras.layers.Embedding(max_features, embedding_dim)(inputs) 
  for i in range(hp.Int('hidden_layers', 1, 3)):
    x = keras.layers.Conv1D(hp.Choice('neuron', values=[100,500,1000,1500,2000])
    , 3, strides=2, padding='same'
    , activation=hp.Choice('activation', values=['relu','sigmoid','tanh']))(x)
  x = keras.layers.GlobalMaxPooling1D()(x)
  x = keras.layers.Dropout(hp.Choice('dropout', values=[0.0,0.2,0.5]))(x)
  predictions = keras.layers.Dense(3, activation='softmax', name='predictions')(x)
  model = tf.keras.Model(inputs, predictions)
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

# randomsearch
t2_model = keras_tuner.GridSearch(
  t2_create_model
  , objective='accuracy'
  , overwrite=True
)

# search
num_epochs = 100
t2_model.search(ks_train_ds, epochs=num_epochs)

# best model results
print("---------------- best params --------------------")
t2_best_param = t2_model.get_best_hyperparameters(num_trials=1)[0]
print("neuron: ", t2_best_param.get("neuron"))
print("activation: ", t2_best_param.get("activation"))
print("hidden_layers: ", t2_best_param.get("hidden_layers"))
print("dropout: ", t2_best_param.get("dropout"))

print("---------------- best model results --------------------")
t2_best_model = t2_model.get_best_models()[0]
t2_best_model.build(ks_train_ds)
print(t2_best_model.summary())

# fit using the best model
t2_best_model.fit(ks_train_ds)

# evaluate on the test dataset
print('----------- Evaluation on Test Dataset ---------------')
test_loss, test_accuracy = t2_best_model.evaluate(ks_test_ds)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# predict/ actual and predict values
y_actual = []
y_predict = []
for text, label in ks_test_ds:
  y_actual.append(label.numpy())
  y_predict.append(t2_best_model.predict(text).argmax(axis=-1))
  
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


