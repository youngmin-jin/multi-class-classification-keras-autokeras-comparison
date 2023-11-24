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

# convert df to tf.dataset with batch 32
target = ['negative', 'neutral', 'positive']

# split
x_train, x_test, y_train, y_test = train_test_split(df['Sentence'], df[target], train_size=0.7)

# ---------------------------------------
# t2-b
# ---------------------------------------
# BertClassifier model 
class t2_b_create_model(keras_tuner.HyperModel):
  def build(self, hp):
    model = keras_nlp.models.BertClassifier.from_preset(
      "bert_large_en_uncased"
      , num_classes=3
      , activation='softmax'
      , dropout=hp.Choice("dropout", values=[0.0,0.2,0.5])
  )
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=hp.Choice("learning_rate", values=[3e-4, 1e-4, 5e-5, 3e-5])), metrics=['accuracy'])
    return model

  def fit(self, hp, model, *args, **kwargs):
    return model.fit(*args, batch_size=hp.Choice("batch_size", values=[8, 16, 32, 64, 128]), **kwargs)

# grid search 
t2_b_model = keras_tuner.GridSearch(
  t2_b_create_model()
  , objective='accuracy'
  , overwrite=True
)

# search
num_epochs = 5
t2_b_model.search(x_train, y_train, epochs=num_epochs)

# best model results
print("---------------- best params --------------------")
t2_b_best_param = t2_b_model.get_best_hyperparameters(num_trials=1)[0]
print("learning_rate: ", t2_b_best_param.get("learning_rate"))
print("dropout: ", t2_b_best_param.get("dropout"))
print("batch_size: ", t2_b_best_param.get("batch_size"))

print("---------------- best model results --------------------")
t2_b_best_model = t2_b_model.get_best_models()[0]
print(t2_b_best_model.summary())

# fit using the best model
t2_b_best_model.fit(x_train, y_train)

# predict/ actual and predict values
print("----------------- y_actual ----------------------")
y_actual = np.argmax(np.array(y_test), axis=1)
print(y_actual)

print("----------------- y_predict ----------------------")
y_predict = t2_b_best_model.predict(x_test)
y_predict = y_predict.argmax(axis=-1)
print(y_predict)

# confusion matrix
print("----------------- confusion matrix ----------------------")
print(confusion_matrix(y_actual, y_predict))

# confusion report
print("----------------- confusion report ----------------------")
print(classification_report(y_actual, y_predict))




