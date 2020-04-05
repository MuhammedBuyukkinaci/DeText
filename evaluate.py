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
TRAIN_ON_KAGGLE = False

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

input_shape = (WIDTH,HEIGHT,CHANNELS)

from model import get_model

model = get_model(compiling=False)

h5_file = [x for x in os.listdir() if 'model_epoch' in x][0]
model = load_model(h5_file)

print(model.summary())

X_val = np.load('data/X_val.npy')
y_val = np.load('data/y_val.npy')
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

val_preds = model.predict(X_val)
test_preds = model.predict(X_test)

val_indexes = list(np.random.choice(len(X_val),20))
test_indexes = list(np.random.choice(len(X_test),20))

for image_no in val_indexes:
    plt.subplot(1, 2, 1)
    plt.imshow(X_val[image_no].astype('uint8'))
    plt.subplot(1, 2, 2)
    plt.imshow(val_preds[image_no].astype('uint8'))
    plt.show()


for image_no in test_indexes:
    plt.subplot(1, 2, 1)
    plt.imshow(X_test[image_no].astype('uint8'))
    plt.subplot(1, 2, 2)
    plt.imshow(test_preds[image_no].astype('uint8'))
    plt.show()