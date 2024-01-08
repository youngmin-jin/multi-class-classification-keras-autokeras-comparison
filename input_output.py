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
# import tensorflow_datasets as tfds

# ----------------------------------------------
# train_size/ necessary variables
# ----------------------------------------------
train_size = 0.7

# - structured data
structured_target = "class"

# - text data
text_target = "Sentiment"
text_predictor = "Sentence"  

batch_size = 32
max_features = 20000
sequency_length = 500
embedding_dim = 128

# ----------------------------------------------
# - structured data & text data
# one hot encoding for categorical columns
# ----------------------------------------------
def create_one_hot_encoding(df, column):
  dummies = pd.get_dummies(df[column])
  encoded_column_name = dummies.columns
  df.drop(column, axis=1, inplace=True)
  df = pd.concat([df, dummies], axis=1)
  return df, encoded_column_name

# ----------------------------------------------
# - structured data & text data
# split to x_train, x_test, y_train, y_test
# ----------------------------------------------
def split_training_test(df, column):
  x_train, x_test, y_train, y_test = train_test_split(df.drop(column, axis=1), df[column], train_size=train_size)
  return x_train, x_test, y_train, y_test

# ----------------------------------------------
# - structured data
# replace column names
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
# - text data
# split to df_train and df_test/ convert them to tf.dataset
# ----------------------------------------------
def split_convert_text_data(df, target_encoded_column_name):
  # split to df_train and df_test
  num = df.shape[0] * train_size
  num = round(num)
  df_train = df.loc[:num]
  df_test = df.loc[num+1:]
  
  # save y_train and y_test for the distributions of classes
  y_train = df_train[target_encoded_column_name]
  y_test = df_test[target_encoded_column_name]
   
  # tf.data.Dataset 
  train_ds = tf.data.Dataset.from_tensor_slices((df_train[text_predictor], df_train[target_encoded_column_name])).batch(batch_size=batch_size)
  test_ds = tf.data.Dataset.from_tensor_slices((df_test[text_predictor], df_test[target_encoded_column_name])).batch(batch_size=batch_size)
  return y_train, y_test, train_ds, test_ds

# ----------------------------------------------
# - o models
# print the best parameters
# ----------------------------------------------
def get_best_params(trained_model, *args):
  print("---------------- best params -----------------")
  best_params = trained_model.get_best_hyperparameters(num_trials=1)[0]
  for arg in args:
    print(arg, ": ", best_params.get(arg)) 

# ----------------------------------------------
# - o models
# return the best model
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
  print("--- y_train ---"), '\n', print(y_train.value_counts())
  print("--- y_test ---"), '\n', print(y_test.value_counts())    

# ----------------------------------------------
# generate confusion matrix
# ----------------------------------------------
def generate_confusion_matrix(trained_model, str_trained_model, x_test, y_test):  
  if str_trained_model.startswith(("Sa","Ia")):
    y_actual = y_test.to_numpy()
    y_predict = trained_model.predict(x_test)
    y_predict = y_predict.flatten()
  
  elif str_trained_model.startswith("Ta"):
    y_actual = y_test
    y_predict = trained_model.predict(x_test)
            
  elif str_trained_model.startswith(("Tm","To")): # Tm, To1, To2
    train_ds = x_test
    test_ds = y_test
    y_actual = []
    y_predict = []
    for text, label in test_ds:
      y_actual.append(label.numpy())
      y_predict.append(trained_model.predict(text).argmax(axis=-1))
    y_actual = np.concatenate(y_actual, axis=0)
    y_actual = np.argmax(np.array(y_actual), axis=1)
    y_predict = np.concatenate(y_predict, axis=0)

  else: # Sm, So, Im, Io1, Io2
    y_actual = np.argmax(np.array(y_test), axis=1)
    y_predict = trained_model.predict(x_test)
    y_predict = y_predict.argmax(axis=-1)

  # print actual values
  print("----------------- y_actual ----------------------")
  print(y_actual)
  
  # print predicted values
  print("----------------- y_predict ----------------------")
  print(y_predict)

  # confusion matrix
  print("----------------- confusion matrix ----------------------")
  print(confusion_matrix(y_actual, y_predict))
    
  # confusion report
  print("----------------- confusion report ----------------------")
  print(classification_report(y_actual, y_predict))


# ==================================================================
# create structured data input
# ==================================================================
def create_structured_input(filepath, flag_one_hot_encoding_on):
  # set a target column and read the data
  target = structured_target
  df = pd.read_csv(filepath, encoding='utf-8')
  
  # rename column names
  renames = create_column_renames(df.columns)
  df.rename(columns=renames, inplace=True)
  
  # return the raw data for automatic models
  if flag_one_hot_encoding_on == False:
    return split_training_test(df, target)
  
  else:
    # one hot encoding for categorical columns
    df, gender_encoded_column_name = create_one_hot_encoding(df, 'gender')
    df, target_encoded_column_name = create_one_hot_encoding(df, target)
    
    # split into training and test datasets
    return split_training_test(df, target_encoded_column_name)
    

# ==================================================================
# create text data input
# ==================================================================
def create_text_input(filepath, flag_bert_on, flag_one_hot_encoding_on):
  # set a target column and read the data
  target = text_target
  df = pd.read_csv(filepath, encoding='utf-8')
  
  # return the raw data for automatic models
  if flag_one_hot_encoding_on == False:
    x_train, x_test, y_train, y_test = split_training_test(df, target)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
  
  else:
    # one hot encoding for categorical columns  
    df, target_encoded_column_name = create_one_hot_encoding(df, target)
    
    if flag_bert_on == True:
      x_train, x_test, y_train, y_test = split_training_test(df, target_encoded_column_name)
      # convert df to series
      x_train = x_train.squeeze()
      return x_train, x_test, y_train, y_test
    
    else:      
      # train and test datasets
      y_train, y_test, train_ds, test_ds = split_convert_text_data(df, target_encoded_column_name)  
      
      # text vectorization
      vectorize_layer = tf.keras.layers.TextVectorization(
          standardize='lower_and_strip_punctuation'
          , max_tokens=max_features
          , output_mode='int'
          , output_sequence_length=sequency_length
      )
      
      # make a text-only dataset
      text_ds = train_ds.map(lambda x,y: x)
      
      # call adapt on a text-only dataset to create the vacabulary
      vectorize_layer.adapt(text_ds)
      
      def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label
      
      # vectorize the data
      train_ds = train_ds.map(vectorize_text)
      test_ds = test_ds.map(vectorize_text)
      
      return y_train, y_test, train_ds, test_ds


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
          
          # convert images into array/ append images and labels to the arrays
          image = img_to_array(image)
          images.append(image)        
          labels.append(class_name)
  
  # convert list into array
  images_array = np.stack(images)
  labels_array = np.array(labels)
  
  # return the raw data for automatic models
  if flag_one_hot_encoding_on == False:
    return train_test_split(images_array, labels_array, train_size=train_size)

  else:
    # split into training and test datasets
    x_train, x_test, y_train, y_test = train_test_split(images_array, labels_array, train_size=train_size)
    
    # data augmentation
    data_augmentation = keras.Sequential(
        [keras.layers.RandomFlip('horizontal')
          , keras.layers.RandomRotation(0.1)
        ]
    )
    
    # apply data augmentation to the training dataset
    x_train = data_augmentation(x_train)
    
    # apply one-hot encoding to target variables
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)
  
    return x_train, x_test, y_train, y_test
