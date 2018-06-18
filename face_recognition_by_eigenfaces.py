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
dirs=os.listdir("C:/Users/garychan/Desktop/facedata/training")
for file in dirs:
    img=cv2.imread("C:/Users/garychan/Desktop/facedata/training/" + file,cv2.IMREAD_GRAYSCALE)
    res1=cv2.resize(img,(100,100))
    res1_1=np.array(res1.reshape(1,10000),dtype=np.int32)
    training_set.append(res1_1)
    
## make a average image
average=np.zeros((1,10000))
average=sum(training_set[:])/500

At=training_set-average
At=At.reshape(500,10000)
A=np.transpose(At)

L=At.dot(A)
w1,v=np.linalg.eig(L)
u=[]
u3=[]
for i in range(500):
    d=np.inner(v[i],A[i])
    u.append(d)

u1=np.argpartition(u,-50)[-50:]   ##u1 contain 50 indices
u2=np.zeros((500,500))            ##u2 contain 50 non-zero tuples
for i in range(50):
    d2=u1[i]
    u2[d2][d2]=u[d2]
    u3.append(u2[d2].dot(At))     ##u3 = 50 eigenfaces here

w=[]
for i in range(500):
    ww=[]
    for j in range(50):
        d=np.inner(u3[j],(training_set[i]-average))
        ww.append(d)
    ww1=np.asarray(ww)
    w.append(ww1.reshape(50,1))

def testing(): 
    TF=0
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
    
def classify(a):
    path="C:/Users/garychan/Desktop/facedata/testing/"+a
    img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    res1=cv2.resize(img,(100,100))
    res1_1=np.array(res1.reshape(1,10000),dtype=np.int32)
    img1=res1_1
    print(path)
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
    plt.imshow((training_set[lll]).reshape(100,100))
    #w.pop()
        #return()
    
#/////////////////////////////////////////////////
#img=cv2.imread("C:/Users/garychan/Desktop/facedata/testing/s18_15.jpg,cv2.IMREAD_GRAYSCALE)
#res1=cv2.resize(img,(100,100))
#res1_1=np.array(res1.reshape(1,10000),dtype=np.int32)
#img1=res1

#hh=[]
#for j in range(50):
#    d=np.inner(u3[j],(res1_1-average))
#    hh.append(d)
#hh1=np.asarray(hh)

        
#for k in range(550):
#   d=np.sqrt(np.sum((hh1-w[k])**2))
#   ll.append(d)
#lll=np.argmin(ll)
#plt.imshow((training_set[lll]).reshape(100,100))
#plt.imshow((res1))

#average=sum/len(training_set)
    
    
    
#//////////////////////////////////////////   
#for x in training_set:
#    res1=cv2.resize(training_set[x],(150,200))
#cv2.imshow('image',training_set[299])
#cv2.waitKey(0)
#cv2.destroyAllWindows()
