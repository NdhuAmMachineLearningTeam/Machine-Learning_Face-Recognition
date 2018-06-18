import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import random

def get_testing_label():                                 #用來隨機產生要歸為測試資料集的index
    testlab = []
    label = np.arange(0,13)
    for i in range(50):                                  #總共50人，每人有13照片，且檔案名稱有次序(一個人13張讀完才會換下一人)
        a = np.random.choice(label, 2, replace=False)    #從13張照片中隨機挑兩張的index出來做測試資料，若2改3則每人挑3張當測試資料
        testlab.append(a)
        label += 13                                      #將random.choice所挑選的母體全部+13，即換到下一個人對應的label
    testlab = np.array(testlab)
    testlab = testlab.reshape(-1)                        #因為前面是兩個兩個添加，是(50,2)的ndarray，這邊做reshape將其轉換成一維
    return testlab

testing_label = get_testing_label()

PATH = "C:/Users/sunny/Desktop/Face Database/"

X_train = []; y_train = []; X_test = []; y_test = []; i = 0;
for file in os.listdir(PATH):                               #讓file依序為PATH目錄下的檔案名稱跑迴圈
    img = mpimg.imread(PATH + file)                         #用imread讀入圖檔至變數img
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        #將圖片轉換為灰度圖
    img_gray = cv2.resize(img_gray,(213,311),interpolation=cv2.INTER_CUBIC)   #將所有圖片通通調成相同的size
    img_gray = img_gray.reshape(-1)                         #轉換為一維陣列方便等下進行學習
    label = int(file[1:3])
    if i in testing_label:                                  #如果index i 屬於前面產生的testing_index集合
        X_test.append(img_gray)                             #就把處理好的圖片與對應label併進測試資料集
        y_test.append(label)
    else:                                                   #否則併入訓練資料集
        X_train.append(img_gray)
        y_train.append(label)
    i += 1
    
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

pca = PCA(n_components=0.97 , svd_solver='full')
newX_train = pca.fit_transform(X_train)
newX_test = pca.transform(X_test)

clf = SVC(kernel = 'linear')
clf.fit(newX_train, y_train)
print(np.mean(clf.predict(newX_train) == y_train))
print(np.mean(clf.predict(newX_test) == y_test))
