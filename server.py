#REST API PREDICTION
import cv2
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
from keras import backend as K
from 
K.clear_session()

def read_img(img_path):
    
    img = cv2.imread(img_path)
    try:
        img = cv2.resize(img, (128,128))
    except:
        return None
    return img

        

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
    #model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    
    #model.add(Conv2D(512, kernel_size=(3, 3), activation= 'relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    #model.add(Conv2D(512, kernel_size=(3, 3), activation= 'relu',kernel_regularizer=regularizers.l2(weight_decay)))
    #model.add(Conv2D(512, kernel_size=(3, 3), activation= 'relu',kernel_regularizer=regularizers.l2(weight_decay)))
    
    model.add(GlobalAveragePooling2D())
    #model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    #model.add(Dropout(0.4))

    
    #model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))
    #model.add(Dense(1024, activation='relu'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    #model.add(Dropout(0.5))        
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())

    return model
    
def load_model(model):
    model.load_weights('new_try5_9.h5')
    global graph
    graph = tf.get_default_graph()

    
       
def test(image_path):
        test_data =[]
        im = read_img(image_path)
        test_data.append(im)
        np.save('test_data_1.npy',test_data)
        test_data = np.load('test_data_1.npy')
        
        model_out = model.predict_classes(test_data)
        #modelout = model.predict_classes(test_data[1:6])
        return model_out[0]
    



from flask import *
import json
from collections import namedtuple
import base64
from PIL import Image

app = Flask(__name__)



class Response:
	def __init__(self,type,fact):
		self.type=type
		self.fact = fact

types =['chocolate cake','Donut','French_fries','Fried rice','Hamburger','Ice-Cream','Omelette','pizza','samosa','spring rolls']
cal=['calories:235,fat:39,carbohydrate:57,protein:4',
'calories:190,fat:49,carbohydrate:45,protein:5'
,'calories:510,fat:43,carbohydrate:52,protein:5',
'calories:329,fat:33,carbohydrate:52,protein:15',
'calories:279,fat:43,carbohydrate:39,protein:18',
'calories:267,fat:46,carbohydrate:47,protein:7',
'calories:290,fat:5,carbohydrate:71,protein:24',
'calories:225,fat:43,carbohydrate:37,protein:20',
'calories:250,fat:32,carbohydrate:55,protein:13',
'calories:63,fat:40,carbohydrate:47,protein:13'
]

@app.route('/upload', methods = ['POST'])
def api_upload():
        x = json.dumps(request.json)
        d = json.loads(x)
        with open("food.jpg", "wb") as fh:
                fh.write(base64.b64decode(d['image']))
        pred = test('food.jpg')
        res = Response(types[pred],cal[pred])
        return json.dumps(res.__dict__)
	
if(__name__=="__main__"):
        global model = bulidModel()

        load_model(model)
        app.run(host="0.0.0.0",debug=True,port=8080)


	
