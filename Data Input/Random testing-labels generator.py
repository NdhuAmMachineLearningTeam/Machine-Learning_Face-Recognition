import numpy as np
import random

def get_testing_label():
    testlab = []
    label = np.arange(0,13)
    for i in range(50):
        a = np.random.choice(label,2)
        testlab.append(a)
        label += 13
    testlab = np.array(testlab)
    testlab = testlab.reshape(-1)
    return testlab
