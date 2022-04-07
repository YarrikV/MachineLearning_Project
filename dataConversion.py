import numpy as np
import random
import glob
import os
import pandas as pd
import cv2
from data.util import *



def reshape(image, s=64):
    """
    Reshapes an image to a y x y resolution. Interpolation method 
    left to default value of linear interpolation.

    :param image: Array of (h,w,3) with h=height and w=width 
    :returns: Array of (y,y,3) 
    """
    return cv2.resize(image, (s, s), interpolation=cv2.INTER_LINEAR)

####    Randomizer
numpy_seed = 0
python_seed = 0

np.random.seed(numpy_seed)
random.seed(python_seed)


####    Load data
root_path = "data"


X, superclass, y, pole, index = TrainDataLoader(root_path).load()
X_test = TestDataLoader(root_path).load()


####    Reshape
X = [reshape(x) for x in X]
X_test = [reshape(x) for x in X_test]


####    Save images as array
filename = os.path.join(root_path, 'resized_train', 'train.npy')
np.save(filename, np.array([X, superclass, y, pole, index]))

filename = os.path.join(root_path, 'resized_train', 'test.npy')
np.save(filename, np.array([X_test]))


# ####    Save image as image
# root_path = "data" 
# for i, img in enumerate(X):
#     filename = os.path.join(root_path, 'resized_train',  f'{i:04d}') + ".png"
    
#     cv2.imwrite(filename, img)
#     if (i+1)%65 == 0:
#         break
    



