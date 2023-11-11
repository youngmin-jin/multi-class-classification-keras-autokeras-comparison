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

# -------------------------------
# version check
# -------------------------------
# from tensorflow import keras

print('tensorflow', tf.__version__)
print('keras', keras.__version__)
print('keras_nlp', keras_nlp.__version__)

# -------------------------------
# read and modify data
# -------------------------------
# read the data
<<<<<<< HEAD
df = pd.read_csv('text-FinancialSentimentAnalysis.csv', encoding='utf-8')
df_original = df.copy()
=======
df = pd.read_csv('text-test.csv', encoding='utf-8')
>>>>>>> db81df7713fac2a61d0c475b51304a913b6bd03b

# convert sentiment to numeric value
def convert_sentiment(value):
  if value == 'positive':
    return 2
  elif value == 'neutral':
    return 1
  else:
    assert value == 'negative'
    return 0

df['numeric_sentiment'] = df.apply(lambda x: convert_sentiment(x['Sentiment']), axis=1)  

<<<<<<< HEAD
# convert df to tf.dataset with batch 32
target = ['negative', 'neutral', 'positive']
=======
# create train test split
x_trainDF, x_testDF, y_trainDF, y_testDF = train_test_split(df[['Sentence']], df[['numeric_sentiment']], test_size= 0.3)


#df_original = df.copy()

## one hot encoding
#dummies = pd.get_dummies(df['Sentiment'])
#df.drop('Sentiment', axis=1, inplace=True)
#df = pd.concat([df, dummies], axis=1)

## split into training and test datasets
#num = df.shape[0] * 0.7
#num = round(num)
#df_train = df.loc[:num]
#df_test = df.loc[num+1:]

## Split into features and target
#target = ['negative', 'neutral', 'positive']
## x = df['Sentence']
## y = df[target]

# Split into training and test sets
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)
>>>>>>> db81df7713fac2a61d0c475b51304a913b6bd03b

#raw_train_ds = tf.data.Dataset.from_tensor_slices((df_train['Sentence'], df_train[target])).batch(batch_size=32)
#raw_test_ds = tf.data.Dataset.from_tensor_slices((df_test['Sentence'], df_test[target])).batch(batch_size=32)

# ---------------------------------------
# t2-b
# ---------------------------------------
# BertClassifier
t2_b_model = keras_nlp.models.BertClassifier.from_preset(
  "bert_tiny_en_uncased"
  , num_classes=3
  , activation='softmax'
)

# without one-hot encoding (but numeric sentiment)
t2_b_model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

# compile
#t2_b_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit
<<<<<<< HEAD
num_epochs = 100
t2_b_model.fit(raw_train_ds, epochs=num_epochs)
=======
num_epochs = 5
# without one-hot encoding
t2_b_model.fit(x=x_trainDF['Sentence'].tolist(), y=y_trainDF['numeric_sentiment'].tolist(), epochs=num_epochs)

#t2_b_model.fit(raw_train_ds, epochs=num_epochs)
>>>>>>> db81df7713fac2a61d0c475b51304a913b6bd03b

# summary
print('----------- results ---------------')
print(t2_b_model.summary())

# predict/ actual and predict values
y_predict = []
# without one-hot encoding
for text in x_testDF['Sentence'].tolist():
  y_predict.append(t2_b_model.predict([text]).argmax(axis=-1))
#y_actual = []
#for text, label in raw_test_ds:
#  y_actual.append(label.numpy())
#  y_predict.append(t2_b_model.predict(text).argmax(axis=-1))
  
#y_actual = np.concatenate(y_actual, axis=0)
#y_actual = np.argmax(np.array(y_actual), axis=1)
y_actual = y_testDF['numeric_sentiment'].to_numpy()
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




