import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
import gc
import cv2
from tqdm import tqdm
tqdm.pandas()
import random
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
TRAIN_ON_KAGGLE = True

import tensorflow as tf
from tensorflow.keras.layers import Input,Conv2D,Lambda,Dropout,MaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose,concatenate
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard
from tensorflow.keras.optimizers import Adam

def seed_everything(seed=51):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
seed_everything(seed=51)

WIDTH = 256
HEIGHT = 256
CHANNELS = 3
#Model Parameters
EPOCHS = 10
BATCH_SIZE = 16

USE_TENSORBOARD = True

input_shape = (WIDTH,HEIGHT,CHANNELS)

from model import get_model

model = get_model(compiling=False)

if TRAIN_ON_KAGGLE:
    h5_file = 'trained_model_kaggle.h5'
    model.load_weights(h5_file)
    adam = Adam(lr = 3e-5)
    model.compile(optimizer = adam, loss = 'mean_squared_error',metrics = [tf.keras.metrics.MeanSquaredError()])

print(model.summary())

input_data = np.load('data/input_data.npy')
target_data = np.load('data/target_data.npy')

X_test = input_data[20000:]
y_test = target_data[20000:]

input_data = input_data[:20000]
target_data = target_data[:20000]

gc.collect()

if TRAIN_ON_KAGGLE == False:
    tf.keras.backend.clear_session()

X_train, X_val, y_train, y_val = train_test_split(input_data, target_data, test_size=0.20, random_state=51)

del input_data,target_data
gc.collect()

if USE_TENSORBOARD:
    checkpoint = ModelCheckpoint('model_epoch_{}.h5'.format(EPOCHS),verbose=1,save_best_only=True)
    early_stopping = EarlyStopping(patience=5,monitor='val_loss')
    tensorboard = TensorBoard(log_dir='logs')
    callbacks = [checkpoint,early_stopping,tensorboard]
else:
    checkpoint = ModelCheckpoint('model_epoch_{}.h5'.format(EPOCHS),verbose=1,save_best_only=True)
    early_stopping = EarlyStopping(patience=5,monitor='val_loss')
    callbacks = [checkpoint,early_stopping]


model.fit(x = X_train, y = y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,\
          verbose=1,validation_data=(X_val,y_val),callbacks=callbacks)

val_preds = model.predict(X_val)

print(val_preds[0].min(),val_preds[0].max())

if False:
    for image_no in range(347,367,1):
    plt.subplot(1, 2, 1)
    plt.imshow(X_val[image_no].astype('uint8'))
    plt.subplot(1, 2, 2)
    plt.imshow(val_preds[image_no].astype('uint8'))
    plt.show()
