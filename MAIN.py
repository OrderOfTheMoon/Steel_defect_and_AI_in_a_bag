# Импорт необходимых зависимостей
import warnings
import numpy as np
import pandas as pd
import math
import os
import itertools
from keras.utils import np_utils
from sklearn.preprocessing import Normalizer, scale
from sklearn.datasets import load_files
import tensorflow as tf
from tensorflow import keras
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from tensorflow.keras.utils import plot_model
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import missingno as msno
#Загрузка датасета
Train = 'Ссылка на тренировочные данные'
Val = 'Ссылка на данные для валидации'
Test = 'Ссылка на данные для тестирования'
#Подготовка датасета
train_datagen = ImageDataGenerator(
    rescale = 1. / 255, #
    rotation_range = 8, #
    zoom_range = 0.1, #
    shear_range = 0.3, #
    width_shift_range = 0.08, #
    height_shift_range = 0.08, #
    vertical_flip = True, #
    horizontal_flip = True) #
test_datagen = ImageDataGenerator(rescale = 1. / 255)
# Flow training images in batches of 10 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = (200, 200),
        batch_size = 10,
        class_mode = 'categorical')
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.98 ): # Stop training the model at 98% traning accuracy
            print("\nReached 98% accuracy so cancelling training!")
            self.model.stop_training = True
# Построение модели по слоям для будущего анализа изображений
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = "relu",
                 strides = 1, padding = "same", data_format = "channels_last"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = "relu",
                 strides = 1, padding = "same", data_format = "channels_last"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = "relu",
                 strides = 1, padding = "same", data_format = "channels_last"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(6, activation = "softmax"))
optimizer = Adam(lr = 0.00002)
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])
model.summary()

# Функция необходимая для отслеживания поведения сети во время обучения
callbacks = myCallback()
history = model.fit(train_generator,
        batch_size = 64,
        epochs = 40,
        validation_data = validation_generator,
        callbacks = [callbacks],
        verbose = 1, shuffle = True)
plt.figure(figsize = (12, 6))
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', label = "Training accuracy")
plt.plot(epochs, val_accuracy, 'r', label = "Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure(figsize = (12, 6))
plt.plot(epochs, loss, 'b', label = "Training loss")
plt.plot(epochs, val_loss, 'r', label = "Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()
test_dir = ' '


def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np.array(data['target'])
    target_labels = np.array(data['target_names'])
    return files, targets, target_labels


x_test, y_test, target_labels = load_dataset(test_dir)
no_of_classes = len(np.unique(y_test))
y_test = np_utils.to_categorical(y_test, no_of_classes)
from keras.preprocessing.image import array_to_img, img_to_array, load_img

def convert_image_to_array(files):
    images_as_array=[]
    for file in files:
        # Convert to Numpy Array
        images_as_array.append(img_to_array(load_img(file)))
    return images_as_array

x_test = np.array(convert_image_to_array(x_test))
print('Test set shape : ',x_test.shape)
x_test = x_test.astype('float32')/255
y_pred = model.predict(x_test)
fig = plt.figure(figsize=(16, 9))
for i, idx in enumerate(np.random.choice(x_test.shape[0], size=16, replace=False)):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_test[idx]))
    pred_idx = np.argmax(y_pred[idx])
    true_idx = np.argmax(y_test[idx])
    ax.set_title("{} ({})".format(target_labels[pred_idx], target_labels[true_idx]),
                 color=("green" if pred_idx == true_idx else "red"))
