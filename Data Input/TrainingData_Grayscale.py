import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2

PATH = "C:/Users/Mr-Fish/Desktop/Face Database/"
X_train = []; y_train = [];
for file in os.listdir(PATH):
    img = mpimg.imread(PATH + file)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X_train.append(img_gray)
    label = int(file[1:3])
    y_train.append(label)
    
for i in range(650):
    X_train[i] = cv2.resize(X_train[i],(213,311),interpolation=cv2.INTER_CUBIC)
    
X_train = np.array(X_train)
y_train = np.array(y_train)

