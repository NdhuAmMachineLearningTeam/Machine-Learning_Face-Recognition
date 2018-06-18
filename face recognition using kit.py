import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2
from sklearn.decomposition import PCA
from sklearn.svm import SVC

PATH = "C:/Users/Mr-Fish/Desktop/Face/"
PATH2 = "C:/Users/Mr-Fish/Desktop/test/"
X_train = []; y_train = []; X_test = []; y_test = [];
for file in os.listdir(PATH):
    img = mpimg.imread(PATH + file)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X_train.append(img_gray)
    label = int(file[1:3])
    y_train.append(label)
    
for i in range(550):
    tem = cv2.resize(X_train[i],(213,311),interpolation=cv2.INTER_CUBIC)
    X_train[i] = tem.reshape(-1)
    
X_train = np.array(X_train)
y_train = np.array(y_train)

for file in os.listdir(PATH2):
    img = mpimg.imread(PATH2 + file)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X_test.append(img_gray)
    label = int(file[1:3])
    y_test.append(label)
    
for i in range(100):
    tem = cv2.resize(X_test[i],(213,311),interpolation=cv2.INTER_CUBIC)
    X_test[i] = tem.reshape(-1)
    
X_test = np.array(X_test)
y_test = np.array(y_test)

pca = PCA(n_components=0.97 , svd_solver='full')
newX_train = pca.fit_transform(X_train)
newX_test = pca.transform(X_test)

clf = SVC(kernel = 'linear')
clf.fit(newX_train, y_train)
print(np.mean(clf.predict(newX_train) == y_train))
print(np.mean(clf.predict(newX_test) == y_test))
