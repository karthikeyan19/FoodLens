import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def bulidModel():
    model= ResNet50(weights=None,input_shape=(384,384,3),classes=30)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    

def train(model):

    # Extracting label columns
    label_cols = list(set(train.columns) - set(['Image_id']))
    label_cols.sort()
    labels = train.iloc[0][2:].index[train.iloc[0][2:] == 1]


    y = train[label_cols].values
    values = np.array(y)

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)



    data = integer_encoded
    data = np.array(data)

    # one hot encode
    encoded = to_categorical(data)


    # Finally, we create the training and validation sets.
    X_train = np.load('train_data.npy') 

    
    Xtrain, Xvalid, ytrain, yvalid = train_test_split(X_train, encoded, test_size=0.33)
    early_stops = EarlyStopping(patience=3, monitor='val_acc')
    checkpointer = ModelCheckpoint(filepath='weights.best.eda.hdf5', verbose=1, save_best_only=True)


    # Finally, we train our model.

    
    model.fit(Xtrain, ytrain, validation_data=(Xvalid, yvalid), epochs=3, batch_size=20, callbacks=[checkpointer], verbose=1)


    
    
#buliding model
model = bulidModel()

#training model
model = train(model)
