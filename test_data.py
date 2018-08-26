import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
import cv2
import os
from random import shuffle

# Reading the train and test meta-data files
TEST_PATH = 'meta/'
TEST_IMG_PATH = "images/"
f = open(TEST_PATH+"new_test.txt","r")

lists = f.readlines()  
print(len(lists))



#TEST_PATH = 'test/'


#img_path = TEST_PATH+str(test.Image_id[0])


from PIL import Image
import cv2


from tqdm import tqdm
def read_img(img_path):
    img = cv2.imread(img_path)
    try:
        img = cv2.resize(img, (128,128))
    except:
        return None
    return img



test_img = []

for img_path in lists:
    p = img_path.replace("\n","")
    im = read_img(TEST_IMG_PATH+p+".jpg")
    if im is not None:
        test_img.append(im)
    


shuffle(test_img)
np.save('test_data_128_10_3.npy', test_img)
