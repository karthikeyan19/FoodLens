import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_files


# Reading the train and test meta-data files
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#train.head()


TRAIN_PATH = 'train/'
TEST_PATH = 'test/'


img_path = TRAIN_PATH+str(train.Image_id[0])


from PIL import Image
import cv2


from tqdm import tqdm
def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (384, 384))
    return img



train_img = []
for img_path in train.Image_id.values:
    train_img.append(read_img(TRAIN_PATH + img_path))



np.save('train_data.npy', train_img)
