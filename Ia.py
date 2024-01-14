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
x_train, x_test, y_train, y_test = create_image_input('Multi-class Weather Dataset', False)

# -------------------------------
# Ia
# -------------------------------
# model
Ia_model = ak.ImageClassifier(num_classes=4, metrics=['accuracy'], overwrite=True, max_trials=2)

# fit
num_epochs = 3
Ia_model.fit(x_train, y_train, epochs=num_epochs)

# summary 
Ia_model_result = Ia_model.export_model()
print(Ia_model_result.summary())

# distributions of classes
distributions_of_classes(y_train, y_test)

# actual and predicted y values/ confusion_matrix
generate_confusion_matrix(Ia_model, "Ia", x_test, y_test)
