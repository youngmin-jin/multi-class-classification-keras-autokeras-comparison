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
x_train, x_test, y_train, y_test = create_structured_input("structured-bodyPerformance.csv", False)

# -------------------------------
# Sa
# -------------------------------
# model
Sa_model = ak.StructuredDataClassifier(
  num_classes=4
  , metrics=['accuracy']
  , overwrite=True
  , max_trials=3
)

# fit
# num_epochs = 100
num_epochs = 5
Sa_model.fit(x_train, y_train, epochs=num_epochs)

# model summary 
Sa_model_result = Sa_model.export_model()
print(Sa_model_result.summary())

# distributions of classes
distributions_of_classes(y_train, y_test)

# actual and predicted y values/ confusion_matrix
generate_confusion_matrix(Sa_model, "Sa_model", x_test, y_test)


# save the best model 
# Sa_model_result.save('Sa_0328_1050.h5', save_format='tf')