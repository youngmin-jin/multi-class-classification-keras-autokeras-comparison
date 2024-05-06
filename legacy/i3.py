# libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import autokeras as ak
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# -------------------------------
# version check
# -------------------------------
import tensorflow as tf
from tensorflow import keras
import keras_tuner
import autokeras as ak

print('tensorflow', tf.__version__)
print('keras', keras.__version__)
print('keras_tuner', keras_tuner.__version__)
print('autokeras', ak.__version__)

# -------------------------------
# read and modify data
# -------------------------------
# generate a dataset
data_dir = 'Multi-class Weather Dataset'
image_size = (180, 180)
images = []
labels = []
class_names = os.listdir(data_dir)

# iterate data implementation using class name
for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)

    for image_file in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_file)
        image = load_img(image_path)
        
        # resize data to form the same shape/ convert them into array
        image = image.resize(image_size)              
        image = img_to_array(image)
        
        # append to the array
        images.append(image)        
        labels.append(class_name)

# convert list into array
images_array = np.stack(images)
labels_array = np.array(labels)

# split
x_train, x_test, y_train, y_test = train_test_split(images_array, labels_array, test_size=0.3)

# -------------------------------
# i3
# -------------------------------
# model
num_epochs = 100
i3_model = ak.ImageClassifier(overwrite=True, num_classes=4, metrics=['accuracy'])
i3_model.fit(x_train, y_train, epochs=num_epochs)

# actual and predict values
y_actual = y_test
y_predict = i3_model.predict(x_test)
y_predict = y_predict.flatten()

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

# summary 
print("---------------- results --------------------")
i3_model_result = i3_model.export_model()
print(i3_model_result.summary())
