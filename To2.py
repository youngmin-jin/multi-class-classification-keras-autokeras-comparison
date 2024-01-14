# libraries 
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import keras_tuner
import keras_nlp
from tensorflow.keras.optimizers import Adam
from input_output import *

# -------------------------------
# version check
# -------------------------------
print('tensorflow', tf.__version__)
print('keras', keras.__version__)
print('keras_tuner', keras_tuner.__version__)
print('keras_nlp', keras_nlp.__version__)

# -------------------------------
# read the data
# -------------------------------
x_train, x_test, y_train, y_test = create_text_input('text-FinancialSentimentAnalysis.csv', True, True)

# ---------------------------------------
# To2
# ---------------------------------------
# model
class To2_create_model(keras_tuner.HyperModel):
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
  
# apply grid search 
To2_model = keras_tuner.GridSearch(
  To2_create_model()
  , objective='accuracy'
  , overwrite=True
  , max_trials=2
)

# search
num_epochs = 3
To2_model.search(x_train, y_train, epochs=num_epochs)

# best parameters
get_best_params(To2_model, "dropout", "learning_rate", "batch_size")

# best model summary
To2_best_model = get_best_model(To2_model, x_train, y_train)

# fit using the best model
To2_best_model.fit(x_train, y_train)

# distributions of classes
distributions_of_classes(y_train, y_test)

# actual and predicted y values/ confusion matrix
generate_confusion_matrix(To2_best_model, "To2_best_model", x_train, y_train)




