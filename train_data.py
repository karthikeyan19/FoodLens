import numpy as np
from sklearn.datasets import load_files
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from PIL import Image
import cv2


# Reading the train data and meta-data files

TRAIN_PATH = 'meta/'
TRAIN_IMG_PATH = "images/"
f = open(TRAIN_PATH+"train.txt","r")

lists = f.readlines()  




def read_img(img_path):
    
    img = cv2.imread(img_path)
    try:
        img = cv2.resize(img, (128,128),interpolation = cv2.INTER_AREA)
    except:
        return None
    return img
    

def train_data_gen():
    train_img = []
    lbl = []
    for img_path in lists: #lists:
        im = read_img(TRAIN_IMG_PATH + img_path.replace("\n","")+".jpg")
        if im is not None:
            name = img_path.split('/')[0]
            lbl.append(name)
            train_img.append(im)



    np.save('train_data_128.npy', train_img)
    np.save('train_data_lbl.npy', lbl)

def one_hot_array_gen():
    lbl = np.load('train_data_lbl.npy')
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(lbl)
    print(integer_encoded)

    data = integer_encoded
    #data = np.array(data)
    # one hot encode
    encoded = to_categorical(data)
    print(encoded[0])
    np.save('train_data_lbl_oha.npy', encoded)

one_hot_array_gen()

