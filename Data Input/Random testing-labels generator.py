import numpy as np
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
