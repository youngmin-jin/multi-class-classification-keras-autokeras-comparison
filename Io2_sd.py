import numpy as np
import tensorflow_datasets as tfds
import autokeras as ak
import tensorflow as tf
import keras_tuner
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from input_output import *

# ------------- var ---------------
image_size = (224, 224)
num_classes = 120


# ------------- data ---------------
# Load the Stanford Dogs dataset
builder = tfds.builder("stanford_dogs")
builder.download_and_prepare()

ds_train = builder.as_dataset(split='train')
ds_test = builder.as_dataset(split='test')

# select only a few
def img_label(example):
  image = example['image']
  label = example['label']
  return image, label

ds_train = ds_train.map(img_label)
ds_test = ds_test.map(img_label)

# preproces
def preprocessing(image, label):
  # resize
  image = tf.image.resize(image, [224, 224])
  # cast images to float
  image = tf.keras.backend.cast(image, dtype=tf.float32)
  # one hot encode the label
  label = tf.one_hot(label, num_classes)

  return image, label

ds_train = ds_train.map(preprocessing).batch(1)
ds_test = ds_test.map(preprocessing).batch(1)


# ------------- model ---------------
# initialize a model
class Io2_create_model(keras_tuner.HyperModel):
  def build(self, hp):  
    # input layer
    inputs = keras.Input(shape=image_size+(3,)) 

    # hidden layers
    x = keras.layers.BatchNormalization()(inputs) 
    x = keras.layers.RandomFlip(hp.Choice("random_flip", values=['horizontal','horizontal_and_vertical']))(x)
    x = keras.applications.EfficientNetB7(
        input_shape=(224,224,3)
        , include_top=False
        , weights='imagenet'
        # , drop_connect_rate=hp.Choice("drop_connect_rate", values=[0.005,0.01,0.1])
        , pooling='avg'
    )(x) 
  
    # output layer
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)  
  
    model = keras.Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=hp.Choice("learning_rate", values=[3e-4,1e-4,5e-5,3e-5,1e-5])), metrics=['accuracy'])
    return model
    
  def fit(self, hp, model, *args, **kwargs):
    return model.fit(*args, batch_size=hp.Choice("batch_size", values=[16,32]), **kwargs)

# apply grid search
Io2_model = keras_tuner.GridSearch(
  Io2_create_model()
  , objective='accuracy'
  , overwrite=True
)

# early stopping
es = keras.callbacks.EarlyStopping(
  monitor="val_accuracy"
  , patience=5
  , restore_best_weights=True
)

# search
num_epochs = 50
Io2_model.search(ds_train, epochs=num_epochs, callbacks=[es])


# ------------- get best/ fit ---------------
# best parameters
get_best_params(Io2_model, "random_flip", "learning_rate", "batch_size")

# best model summary 
Io2_best_model = get_best_model(Io2_model, ds_train)

# fit using the best model
Io2_best_model.fit(ds_train)


# ------------- results ---------------
# confusion_matrix, classification report
y_actual = []
for image, label in ds_test:
  y_actual.append(np.argmax(label.numpy()))
y_actual = np.array(y_actual)

pred = Io2_best_model.predict(ds_test)
y_pred = np.argmax(pred, axis=1)

print("---- confusion matrix ----")
print(confusion_matrix(y_actual, y_pred))

print("---- classification report ----")
print(classification_report(y_actual, y_pred))



