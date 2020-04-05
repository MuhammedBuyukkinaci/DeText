import tensorflow as tf
from tensorflow.keras.layers import Input,Conv2D,Lambda,Dropout,MaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose,concatenate
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
WIDTH = 256
HEIGHT = 256
CHANNELS = 3
input_shape = (WIDTH,HEIGHT,CHANNELS)

def get_model(input_shape = input_shape,compiling=True):
    """
    Defining a Unet Architecture
    """
    ##Contraction Path##
    #Input
    inputs = Input(input_shape)
    #Lambda
    scaled = tf.keras.layers.Lambda(lambda x: x/255)(inputs)
    #Conv1
    conv1 = Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(scaled)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv1)
    pool1 = MaxPooling2D((2,2))(conv1)
    #Conv2
    conv2 = Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv2)
    pool2 = MaxPooling2D((2,2))(conv2)
    #Conv3
    conv3 = Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv3)
    pool3 = MaxPooling2D((2,2))(conv3)
    #Conv4
    conv4 = Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv4)
    pool4 = MaxPooling2D((2,2))(conv4)
    #Conv5
    conv5 = Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(pool4)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv5)
    ##Expansive Path##

    #Conv6
    conv6_up = Conv2DTranspose(128,(2,2),strides = (2,2),padding='same')(conv5)
    conv6_up = concatenate([conv6_up,conv4])
    conv6 = Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv6_up)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv6)

    #Conv7
    conv7_up = Conv2DTranspose(64,(2,2),strides = (2,2),padding='same')(conv6)
    conv7_up = concatenate([conv7_up,conv3])
    conv7 = Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv7_up)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv7)

    #Conv8
    conv8_up = Conv2DTranspose(32,(2,2),strides = (2,2),padding='same')(conv7)
    conv8_up = concatenate([conv8_up,conv2])
    conv8 = Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv8_up)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv8)

    #Conv9
    conv9_up = Conv2DTranspose(16,(2,2),strides = (2,2),padding='same')(conv8)
    conv9_up = concatenate([conv9_up,conv1])
    conv9 = Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv9_up)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(conv9)

    #Output Layer
    outputs = Conv2D(3,(1,1),activation='sigmoid')(conv9)

    #Lambda
    outputs_scaled = tf.keras.layers.Lambda(lambda x: x*255)(outputs)

    #Model
    adam = Adam(lr = 3e-4)
    model = Model(inputs= [inputs],outputs = [outputs_scaled])
    if compiling:
        model.compile(optimizer = adam, loss = 'mean_squared_error',metrics = [tf.keras.metrics.MeanSquaredError()])
    return model