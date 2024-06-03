# libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# ----------------------------------------------
# variables
# ----------------------------------------------
# train size
train_size = 0.7

# structured data
structured_target = 'class'
structured_categorical_col = 'gender'

# text data
text_target = 'Sentiment'

batch_size = 32
max_features = 20000
sequency_length = 500
embedding_dim = 128

# image data
image_size = (180, 180)

# ----------------------------------------------
# structured data & text data
# - one hot encoding for categorical columns
# ----------------------------------------------
def create_one_hot_encoding(df, column):
  dummies = pd.get_dummies(df[column])
  encoded_column_name = dummies.columns
  df.drop(column, axis=1, inplace=True)
  df = pd.concat([df, dummies], axis=1)
  return df, encoded_column_name

# ----------------------------------------------
# structured data & text data
# - split to x_train, X_test, y_train, y_test
# ----------------------------------------------
def split_training_test(df, target):
  X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1), df[target], train_size=train_size)
  return X_train, X_test, y_train, y_test

# ----------------------------------------------
# structured data
# - replace column names
# ----------------------------------------------
def create_column_renames(columns):
  renames = {}
  for column in columns:
    new_column = column
    new_column = new_column.replace('%', 'per')
    new_column = new_column.replace(' ', '_')
    new_column = new_column.replace('-', '_')
    renames[column] = new_column
  return renames

# ----------------------------------------------
# o models
# - print the best hyperparameters
# ----------------------------------------------
def get_best_params(trained_model, *args):
  print("---------------- best params -----------------")
  best_params = trained_model.get_best_hyperparameters(num_trials=1)[0]
  for arg in args:
    print(arg, ": ", best_params.get(arg)) 

# ----------------------------------------------
# o models
# - return the best model
# ----------------------------------------------
def get_best_model(trained_model, *args):
  print("------------ best model summary -------------")
  best_model = trained_model.get_best_models()[0]
  best_model.build(args)
  print(best_model.summary())
  return best_model

# ----------------------------------------------
# distributions of the classes
# ----------------------------------------------
def distributions_of_classes(y_train, y_test):
  print("-------------- distribution of classes ------------")
  print("--- y_train ---"), '\n'
  if isinstance(y_train, np.ndarray):
    unique, counts = np.unique(y_train, return_counts=True)
    print(np.asarray((unique, counts)).T)
  else:
    print(y_train.value_counts())    

  print("--- y_test ---"), '\n'
  if isinstance(y_test, np.ndarray):
    unique, counts = np.unique(y_test, return_counts=True)
    print(np.asarray((unique, counts)).T)
  else:
    print(y_test.value_counts())         

# ----------------------------------------------
# generate confusion matrix
# ----------------------------------------------
def generate_confusion_matrix(trained_model, str_trained_model, X_test, y_test):  
  if str_trained_model.startswith(("Sa","Ia","Ta")):
    if str_trained_model.startswith("Sa"):
      y_actual = y_test.to_numpy()
    else:
      y_actual = y_test
    y_predict = trained_model.predict(X_test)
    y_predict = y_predict.flatten()

  else: 
    y_actual = y_test.values.argmax(axis=1)
    y_predict = trained_model.predict(X_test)
    y_predict = y_predict.argmax(axis=-1)

  print("----------------- y_actual ----------------------")
  print(y_actual)  
  print("----------------- y_predict ----------------------")
  print(y_predict)
  print("----------------- confusion matrix ----------------------")
  print(confusion_matrix(y_actual, y_predict))
  print("----------------- confusion report ----------------------")
  print(classification_report(y_actual, y_predict))


# ==================================================================
# create structured data input
# ==================================================================
def create_structured_input(filepath, flag_one_hot_encoding_on):
  # read the data
  df = pd.read_csv(filepath, encoding='utf-8')
  
  # rename column names
  renames = create_column_renames(df.columns)
  df.rename(columns=renames, inplace=True)
  
  # return the raw data for automatic models
  if flag_one_hot_encoding_on == False:
    return split_training_test(df, structured_target)
  
  else:
    # apply one-hot encoding to categorical column and target column
    df, gender_encoded_column_name = create_one_hot_encoding(df, structured_categorical_col)
    df, target_encoded_column_name = create_one_hot_encoding(df, structured_target)
    
    # return the one-hot encoded data for minimal and optimized models
    return split_training_test(df, target_encoded_column_name)
    

# ==================================================================
# create text data input
# ==================================================================
def create_text_input(filepath, flag_one_hot_encoding_on, flag_bert_on):
  # read the data
  df = pd.read_csv(filepath, encoding='utf-8')
  
  if flag_bert_on == True:
    return split_training_test(df, text_target)
  
  # return the data converted to np.array for automatic models
  if flag_one_hot_encoding_on == False:
    X_train, X_test, y_train, y_test = split_training_test(df, text_target)
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
  
  else:
    # apply one hot encoding to the target column 
    df, target_encoded_column_name = create_one_hot_encoding(df, text_target)
    X_train, X_test, y_train, y_test = split_training_test(df, target_encoded_column_name)
    
    # return the data converted to series for an optimized model (using BERT) 
    if flag_bert_on == True:   
      X_train = X_train.squeeze()
      X_test = X_test.squeeze()
      return X_train, X_test, y_train, y_test

    else:      
      # text vectorization
      vectorize_layer = tf.keras.layers.TextVectorization(
          standardize='lower_and_strip_punctuation'
          , max_tokens=max_features
          , output_mode='int'
          , output_sequence_length=sequency_length
      )
      vectorize_layer.adapt(X_train)      
      
      X_train  = vectorize_layer(X_train)      
      X_test  = vectorize_layer(X_test)
      
      
      # one-hot encode y labels
      y_train = pd.get_dummies(y_train)
      y_test = pd.get_dummies(y_test)
        
      print("--------------- X_train y_train ----------------")
      print(X_train)
      print(y_train)
      
      return X_train, X_test, y_train, y_test


# ==================================================================
# create image data input
# ==================================================================
def create_image_input(filepath, flag_one_hot_encoding_on):
  images = []
  labels = []
  class_names = os.listdir(filepath)
  
  # iterate the implementation using class name
  for class_name in class_names:
      class_dir = os.path.join(filepath, class_name)
  
      for image_file in os.listdir(class_dir):
          image_path = os.path.join(class_dir, image_file)
          image = load_img(image_path)
          
          # resize data to form the same shape/ convert images into array
          image = image.resize(image_size)
          image = img_to_array(image)
                    
          # append images and labels to the arrays
          images.append(image)        
          labels.append(class_name)
  
  # convert list into array
  images_array = np.stack(images)
  labels_array = np.array(labels)
  
  # return the data converted to array for automatic models
  if flag_one_hot_encoding_on == False:
    return train_test_split(images_array, labels_array, train_size=train_size)

  else:
    # split into training and test datasets
    X_train, X_test, y_train, y_test = train_test_split(images_array, labels_array, train_size=train_size)
    
    # data augmentation
    data_augmentation = keras.Sequential(
        [keras.layers.RandomFlip('horizontal')
          , keras.layers.RandomRotation(0.1)
        ]
    )
    
    # apply data augmentation to the training dataset
    X_train = data_augmentation(X_train)
    
    # apply one-hot encoding to the target column
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)
  
    return X_train, X_test, y_train, y_test
