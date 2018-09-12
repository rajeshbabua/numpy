
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy import *
%matplotlib inline

arr = np.array([[255,0,0,0,0,0,0,0],[0,255,0,0,0,0,0,0],[0,0,255,0,0,0,0,0],[0,0,0,255,0,0,0,0],[0,0,0,0,255,0,0,0],[0,0,0,0,0,255,0,0],
               [0,0,0,0,0,0,255,0],[0,0,0,0,0,0,0,255]],dtype= np.int32)
arr.shape
arr.ndim

im = Image.fromarray(arr)
im.show()
