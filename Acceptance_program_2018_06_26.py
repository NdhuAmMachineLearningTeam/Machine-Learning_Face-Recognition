import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import random

PATH1 = input("輸入訓練資料集所在的路徑位址: ") + "\\"
PATH2 = input("輸入測試資料集所在的路徑位址: ") + "\\"

X_train = []; y_train = []; X_test = []; y_test = [];
for file in os.listdir(PATH1):                               #讓file依序為PATH目錄下的檔案名稱跑迴圈
    img = mpimg.imread(PATH1 + file)                         #用imread讀入圖檔至變數img
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)         #將圖片轉換為灰度圖
    img_gray = cv2.resize(img_gray,(213,311),interpolation=cv2.INTER_CUBIC)   #將所有圖片通通調成相同的size
    img_gray = img_gray.reshape(-1)                          #轉換為一維陣列方便等下進行學習
    label = int(file[1:3])
    X_train.append(img_gray)                                 #把處理好的圖片與對應label併進訓練資料集
    y_train.append(label)
for file in os.listdir(PATH2):
    img = mpimg.imread(PATH2 + file)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray,(213,311),interpolation=cv2.INTER_CUBIC)
    img_gray = img_gray.reshape(-1)
    label = int(file[1:3])
    X_test.append(img_gray)                                  #把處理好的圖片與對應label併進測試資料集
    y_test.append(label)
    
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

pca = PCA(n_components=0.97 , svd_solver='full')
newX_train = pca.fit_transform(X_train)
newX_test = pca.transform(X_test)

clf = SVC(kernel = 'linear')
clf.fit(newX_train, y_train)
print("訓練資料辨識率:",np.mean(clf.predict(newX_train) == y_train))
print("測試資料辨識率:",np.mean(clf.predict(newX_test) == y_test))

def get_display_label(n):
    displaylabel = []
    labelrange = np.arange(len(y_test))
    displaylabel = np.random.choice(labelrange, n, replace=False)
    return displaylabel

n = int(input("Enter the number of results you want to display: "))
display_index = get_display_label(n)
for i in display_index:
    img = X_test[i].reshape((311,213))
    plt.imshow(img, cmap='Greys_r')
    plt.axis('off')
    plt.show()
    print("辨識結果:",clf.predict(newX_test[i].reshape(1, -1)))
    print("\n實際ID:",y_test[i])
