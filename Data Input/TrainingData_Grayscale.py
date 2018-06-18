import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2

PATH = "C:/Users/Mr-Fish/Desktop/Face Database/"
X_train = []; y_train = [];
for file in os.listdir(PATH):                               #讓file依序為PATH目錄下的檔案名稱跑迴圈
    img = mpimg.imread(PATH + file)                         #用imread讀入圖檔至變數img
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        #將圖片轉換為灰度圖
    img_gray = cv2.resize(img_gray,(213,311),interpolation=cv2.INTER_CUBIC)   #將所有圖片通通調成相同的size
    img_gray = img_gray.reshape(-1)                         #轉換為一維陣列方便等下進行學習
    X_train.append(img_gray)
    label = int(file[1:3])                                  #取檔案的第二第三個字元轉換成數字作為監督式學習的Label
    y_train.append(label)
    
X_train = np.array(X_train)
y_train = np.array(y_train)
