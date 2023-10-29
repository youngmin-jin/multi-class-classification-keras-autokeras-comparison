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
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# -------------------------------
# read and modify data
# -------------------------------
# generate a dataset
data_dir = 'Multi-class Weather Dataset'
image_size = (128, 128)
images = []
labels = []
class_names = os.listdir(data_dir)
num_classes = len(class_names)
label_index = {class_name: idx for idx, class_name in enumerate(class_names)}

for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)

    for image_file in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_file)
        image = load_img(image_path)
        image = image.resize(image_size)
        image = img_to_array(image)
        images.append(image)
        
        labels.append(label_index[class_name])

images_array = np.stack(images)
labels_array = np.array(labels)

for i in range(len(images_array)):
  images_array[i] = np.reshape(images_array[i], image_size + (3,))

print("------ images_array shape -----", images_array.shape)
print("------ images_array type -----", type(images_array))
print("------ images_array dtype -----", images_array.dtype)

print("------ labels_array shape -----", labels_array.shape)
print("------ labels_array type -----", type(labels_array))
print("------ labels_array dtype -----", labels_array.dtype)

# split
x_train, x_test, y_train, y_test = train_test_split(images_array, labels_array, test_size=0.3)

print("------ x_train shape dtype -----", x_train.shape, x_train.dtype)
print("------ y_train shape dtype -----", y_train.shape, y_train.dtype)
print("------ x_test shape dtype -----", x_test.shape, x_test.dtype)
print("------ y_test shape dtype -----", y_test.shape, y_test.dtype)


# -------------------------------
# i3
# -------------------------------
# model
num_epochs = 5
i3_model = ak.ImageClassifier(overwrite=True, num_classes=4, max_trials=2, metrics=['accuracy'])
i3_model.fit(x_train, y_train, epochs=num_epochs)

# evaluate on the test dataset
# print('----------- Evaluation on Test Dataset ---------------')
# test_loss, test_accuracy = i3_model.evaluate(x_test)
# print(f'Test Loss: {test_loss:.4f}')
# print(f'Test Accuracy: {test_accuracy:.4f}')

# predict/ actual and predict values
y_actual = y_test

y_predict = i3_model.predict(x_test)
y_predict = y_predict.flatten()
y_predict = y_predict.astype('int')

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
