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
    weight_decay = 1e-4
    model = Sequential()
    model.add(BatchNormalization(input_shape=(128,128,3)))
    model.add(Conv2D(8, kernel_size=(3, 3), activation= 'relu',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(16, kernel_size=(3, 3), activation= 'relu',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(32, kernel_size=(3, 3), activation= 'relu',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(64, kernel_size=(3, 3), activation= 'relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation= 'relu',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, kernel_size=(3, 3), activation= 'relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation= 'relu',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, kernel_size=(3, 3), activation= 'relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    #model.add(Conv2D(256, kernel_size=(3, 3), activation= 'relu',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
        
    model.add(Dense(101))
    model.add(Activation('softmax'))
    #from keras import backend as K
    #model.add(Lambda(lambda x: K.tf.nn.softmax(x)))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    
    return model
    
def load_model(model):
    model.load_weights('weights.best.eda.hdf5')

def train(model):
    batch_size = 16

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            #rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    #test_datagen = ImageDataGenerator(rescale=1./255)

    x = np.load('train_data_128.npy')
              
    # Finally, we create the training and validation sets.
    #x=x/255

    #mean = x.mean(axis=(0,1,2,3))
    #std = x.std(axis=(0,1,2,3))
    #x=(x-mean)/(std+le-7)          
              
    y = np.load('train_data_lbl_oha.npy')
    #y = np.array(y)
    Xtrain, Xvalid, ytrain, yvalid = train_test_split(x, y, test_size=0.33)
    model.fit_generator(
            train_datagen.flow(Xtrain,ytrain,batch_size=16),
            steps_per_epoch=(66667 // batch_size),
            epochs=5,
            validation_data=(Xvalid,yvalid),
            validation_steps=(33333 // batch_size))
    model.save_weights('first_try.h5')  # always save your weights after training or during training//

  
    #early_stops = EarlyStopping(patience=3, monitor='val_acc')
    #checkpointer = ModelCheckpoint(filepath='weights.best.eda.hdf5', verbose=1, save_best_only=True)


    # Finally, we train our model.

    
    #model.fit(Xtrain, ytrain, validation_data=(Xvalid, yvalid), epochs=3, batch_size=1, callbacks=[checkpointer], verbose=1)

def test_data(model):
    test_data = np.load('test_data.npy')
    fig=plt.figure()
    d = np.load("lables.npy")
    # one hot encode
    model_out = model.predict(test_data[:12])
        

    for i in range(12):
        
        #img_num = data[1]
        img_data = test_data[i]
        
        y = fig.add_subplot(3,4,i+1)
        orig = img_data
        
        y.imshow(orig)
        plt.title(d[np.argmax(model_out[i])])
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()



f=h5py.File('food_c101_n1000_r384x384x3.h5','r')  


y=np.array([[int(i) for i in f["category"][j]] for j in range(len(f["category"]))])
    

print(y[0])    
    
###buliding model
model = bulidModel()
##
###training model
model = train(model)
##
###loading from saved model
##load_model(model)
##
###testing data
##test_data(model)
##
##
