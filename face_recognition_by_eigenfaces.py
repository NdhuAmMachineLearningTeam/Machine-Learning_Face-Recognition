# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 06:24:58 2018

@author: garychan
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

training_set=[]
testing_set=[]

## import all the training data
dirs=os.listdir("C:/Users/garychan/Desktop/facedata/training")      #讀入訓練圖張，每人各十張
for file in dirs:
    img=cv2.imread("C:/Users/garychan/Desktop/facedata/training/" + file,cv2.IMREAD_GRAYSCALE)
    res1=cv2.resize(img,(100,100))
    res1_1=np.array(res1.reshape(1,10000),dtype=np.int32)
    training_set.append(res1_1)
    
## make a average image
average=np.zeros((1,10000))                                          #把500張人臉合成為一張平均臉
average=sum(training_set[:])/500

At=training_set-average                                              #At-average是要讓每張人臉把所有人都有的特徵去掉，比如說頭髮，
At=At.reshape(500,10000)                                             #這種是比較集中且重覆的特徵
A=np.transpose(At)

L=At.dot(A)                                                           #L就是要開始做降維度的動作
w1,v=np.linalg.eig(L)                                                 #10000x10000要算eigenvalue會跑不完
u=[]                                                                  #但是因為訓練圖片只有500張，500x500矩陣的eigenvalue不用等就能直接算出來
u3=[]                                                                 #所以我就直接用500x500沒有再額外做降維
for i in range(500):
    d=np.inner(v[i],A[i])
    u.append(d)

u1=np.argpartition(u,-50)[-50:]   ##u1 contain 50 indices             #u, u1, u2, u3 都是在抽取特徵
u2=np.zeros((500,500))            ##u2 contain 50 non-zero tuples     #這邊我是用250000個特徵中選了50個，取法是250000個點中的最大的50個點
for i in range(50):                                                   #取50是因為我覺得有50個人
    d2=u1[i]                                                          #但事實上發現改成30測試是99/150, 改70測試是100/150
    u2[d2][d2]=u[d2]                                                  #所以其實30也夠
    u3.append(u2[d2].dot(At))     ##u3 = 50 eigenfaces here

w=[]                                                                  #w是把我取的500張訓練集裡面的臉通通送到eigenspace(1x50維)去
for i in range(500):                                                  
    ww=[]
    for j in range(50):
        d=np.inner(u3[j],(training_set[i]-average))
        ww.append(d)
    ww1=np.asarray(ww)
    w.append(ww1.reshape(50,1))

def testing():                                                         #testing是一個一鍵直接把150張測試圖片自己吃進去，然後告訴你
    TF=0                                                               #哪張辨錯，哪張辨對，再把辨識正確的數字印出來，這邊結果是100/150
    a=["01","02","03","04","05","06","07","08","09","10","11","12","13","14"
       ,"15","16","17","18","19","20","21","22","23","24","25","26","27","28"
       ,"29","30","31","32","33","34","35","36","37","38","39","40","41","42"
       ,"43","44","45","46","47","48","49","50"]
    b=["13","14","15"]
    for x in range(50):
        path1="C:/Users/garychan/Desktop/facedata/testing/s"+a[x]
        for y in range(3):
            path=path1+"_"+b[y]+".jpg"
            img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            print(path)
            res1=cv2.resize(img,(100,100))
            res1_1=np.array(res1.reshape(1,10000),dtype=np.int32)
            img1=res1_1               
        #print(img1)
            hh=[]
            for k in range(50):
                d=np.inner(u3[k],(img1-average))
                hh.append(d)
                hhh=np.asarray(hh)
                #w.append(np.asarray(hhh))
                #print(hh)
            ll=[]
            for k in range(500):
                d=np.sqrt(np.sum(np.square(hhh-w[k])))
                ll.append(d)
            lll=np.argmin(ll)
            print(lll)
            if ((x+1)*10-11<lll<(x+1)*10):
                TF = TF+1
                print("test"+a[x]+"_"+b[y]+" is true.")
            else :
                continue
    print("number of correct guesses",TF)
            #plt.imshow((training_set[lll]).reshape(100,100))
            #w.pop()
    
def classify(a):                                                          #classify()就是辨識器的function，a要輸入完整的圖片名稱
    path="C:/Users/garychan/Desktop/facedata/testing/"+a                  #e.g. s14_11.jpg
    img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    res1=cv2.resize(img,(100,100))
    res1_1=np.array(res1.reshape(1,10000),dtype=np.int32)
    img1=res1_1
    print(path)
    #print(img1)
    hh=[]
    for k in range(50):                                                    #把圖片降階到eigen space去跟w做比較
        d=np.inner(u3[k],(img1-average))
        hh.append(d)
        hhh=np.asarray(hh)
        #w.append(np.asarray(hhh))
        #print(hh)
        ll=[]
    for k in range(500):                                                   #辨識的方法是跟w集合的歐氏距離做比較，選最靠近他的
        d=np.sqrt(np.sum(np.square(hhh-w[k])))
        ll.append(d)
    lll=np.argmin(ll)
    print(lll)
    plt.imshow((training_set[lll]).reshape(100,100))
    #w.pop()
        #return()
    

#//////////////////////////////////////////////////////////////////////////////
#總結
#1. 缺點
#   a. 辨識率不高
#   b. 有些人從來沒有被辨識出來(估計因為是在做eigenface的時候，有些人的一個特徵都沒有被取出來)
#   c. python初心者，語法可能不太行
#2. 改進的方法
#   a. 選擇更有特色的特徵點
#   b. 分辨不要用歐式距離，可以改用KNN,decision tree
#   c. 方法的局限性，因為是取特徵，所以照片背景的光亮度跟人臉有沒有正對鏡頭影響很大，用更正規的圖片辨識率應該會更好一點
