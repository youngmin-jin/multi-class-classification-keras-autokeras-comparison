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


# -------------------------------
# replacing column names for structured data
# -------------------------------
def create_column_renames(columns):
  renames = {}
  for column in columns:
    new_column = column
    new_column = new_column.replace('%', 'per')
    new_column = new_column.replace(' ', '_')
    new_column = new_column.replace('-', '_')
    renames[column] = new_column
  return renames

# -------------------------------
# one hot encoding for structured data
# -------------------------------
def create_one_hot_encoding(df, columns):
  for column in columns:
    dummies = pd.get_dummies(df[column])
    df.drop(column, axis=1, inplace=True)
    df = pd.concat([df, dummies], axis=1)
  return df


# -------------------------------
# create structured data input
# -------------------------------
def create_structured_input(filepath, flag_one_hot_encoding_on):
  # read the data
  df = pd.read_csv(filepath, encoding='utf-8')
  
  # rename column names
  renames = create_column_renames(df.columns)
  df.rename(columns=renames, inplace=True)
  
  if flag_one_hot_encoding_on == True:
    return train_test_split(df.drop('class', axis=1), df[['class']], train_size= 0.7)
  
  else:
    # one hot encoding
    df = create_one_hot_encoding(df, ['gender','class'])
    
    # split into training and test datasets
    target = ['A','B','C','D']
    predictors = np.setdiff1d(df.columns.to_numpy(), target)
    return train_test_split(df[predictors], df[target], train_size=0.7) 
    

# -------------------------------
# create text data input
# -------------------------------
def create_text_input(filepath, flag_one_hot_encoding_on):
  # read the data
  df = pd.read_csv(filepath, encoding='utf-8')
  
  if flag_one_hot_encoding_on == True:
    x_train, x_test, y_train, y_test = train_test_split(df[['Sentence']], df[['Sentiment']], train_size= 0.7)  
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
  
  else:
    # one hot encoding    
    df = create_one_hot_encoding(df, ['Sentiment'])
    
    # prepare for the splitting
    num = df.shape[0] * 0.7
    num = round(num)
    df_train = df.loc[:num]
    df_test = df.loc[num+1:]
    
    # convert df to tf.dataset
    target = ['negative','neutral','positive']
    train_ds = tf.data.Dataset.from_tensor_slices((df_train['Sentence'], df_train[target]))
    test_ds = tf.data.Dataset.from_tensor_slices((df_test['Sentence'], df_test[target]))
    
    # split train, test datasets
    x_train = train_ds.map(lambda x,y: x)
    y_train = train_ds.map(lambda x,y: y)
    x_test = test_ds.map(lambda x,y: x)
    y_test = test_ds.map(lambda x,y: y)
    
    # prepare the setting
    max_features = 20000
    embedding_dim = 128
    sequency_length = 500
    
    vectorize_layer = tf.keras.layers.TextVectorization(
        standardize='lower_and_strip_punctuation'
        , max_tokens=max_features
        , output_mode='int'
        , output_sequence_length=sequency_length
    )
    
    # call adapt on a text-only dataset to create the vacabulary
    vectorize_layer.adapt(x_train)
    
    def vectorize_text(text):
      text = tf.expand_dims(text, -1)
      return vectorize_layer(text)
    
    # vectorize the data
    x_train = x_train.map(vectorize_text)
    x_test = x_test.map(vectorize_text)
    
    return x_train, x_test, y_train, y_test

# -------------------------------
# create image data input
# -------------------------------
def create_image_input(filepath, flag_one_hot_encoding_on):
  # preparation for implementing image data
  data_dir = filepath
  image_size = (180, 180)
  images = []
  labels = []
  class_names = os.listdir(data_dir)
  
  # iterate the implementation using class name
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
  
  if flag_one_hot_encoding_on == True:
    return train_test_split(images_array, labels_array, train_size=0.7)

  else:
    # split
    x_train, x_test, y_train, y_test = train_test_split(images_array, labels_array, test_size=0.3)
    
    # data augmentation model
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


# -------------------------------
# generate confusion matrix
# -------------------------------
def generate_confusion_matrix(trained_model, x_test, y_test):
  # print actual and predicted y values
  print("----------------- y_actual ----------------------")
  y_actual = np.argmax(np.array(y_test), axis=1)
  print(y_actual)
  
  print("----------------- y_predict ---------------------")
  y_predict = trained_model.predict(x_test)
  y_predict = y_predict.argmax(axis=-1)
  print(y_predict)
  
  # confusion matrix
  print("----------------- confusion matrix ----------------------")
  print(confusion_matrix(y_actual, y_predict))
  
  # confusion report
  print("----------------- confusion report ----------------------")
  print(classification_report(y_actual, y_predict))


