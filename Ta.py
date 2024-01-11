# libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import keras_tuner
import autokeras as ak
from input_output import *

# -------------------------------
# version check
# -------------------------------
print('tensorflow', tf.__version__)
print('keras', keras.__version__)
print('keras_tuner', keras_tuner.__version__)
print('autokeras', ak.__version__)

# -------------------------------
# read the data
# -------------------------------
x_train, x_test, y_train, y_test = create_text_input('text-FinancialSentimentAnalysis.csv', False, False)

# -------------------------------
# Ta
# -------------------------------
# model
Ta_model = ak.TextClassifier(num_classes=3, metrics=['accuracy'], overwrite=True, max_trials=2)

# fit
num_epochs = 3
Ta_model.fit(x_train, y_train, epochs=num_epochs)

# summary 
Ta_model_result = Ta_model.export_model()
print(Ta_model_result.summary())

# distributions of classes
distributions_of_classes(y_train, y_test)

# actual and predicted y values/ confusion_matrix
generate_confusion_matrix(Ta_model, "Ta_model", x_test, y_test)
