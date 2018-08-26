import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from glob import glob
import h5py
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras import optimizers
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D,Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization,Activation,Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers

def bulidModel():
    
    weight_decay = 1e-3
    model = Sequential()
    model.add(BatchNormalization(input_shape=(128,128,3)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation= 'relu',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation= 'relu',kernel_regularizer=regularizers.l2(weight_decay)))
 
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

  
    model.add(Conv2D(128, kernel_size=(3, 3), activation= 'relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation= 'relu',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    model.add(Conv2D(256, kernel_size=(3, 3), activation= 'relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation= 'relu',kernel_regularizer=regularizers.l2(weight_decay)))
   
    model.add(GlobalAveragePooling2D())
    
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
     
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    return model
    
def load_model(model):
    model.load_weights('new_try5_9.h5')

def train(model):
    batch_size = 16


    x = np.load('train_data_128_10_3.npy')
              
  
    y = np.load('train_data_lbl_oha3.npy')
    Xtrain, Xvalid, ytrain, yvalid = train_test_split(x, y, test_size=0.30)


    #  train our model.

    

    model.fit(Xtrain, ytrain, validation_data=(Xvalid, yvalid), epochs=50, batch_size=16,  verbose=1)
    model.save_weights('new_try5_10.h5')  # always save your weights after training 
def test_data(model):
    test_data = np.load('test_data_128_10_3.npy')
    fig=plt.figure()
    fig=plt.plot()
    d = ['Cake','Donuts','fries','rice','hamburger','ice_cream','omelette','pizza','samosa','rolls']
    # one hot encode
    model_out = model.predict_classes(test_data)
    
    for i in range(12):
        
        
        img_data = test_data[i+1000]
        
        y = fig.add_subplot(3,4,i+1)
        orig = img_data
        
        y.imshow(orig)
        plt.title(d[np.argmax(model_out[i])])
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()



###buliding model
model = bulidModel()
##
load_model(model)

###training model
model = train(model)
##
###loading from saved model
##
###testing data
test_data(model)
##
##

