import pandas as pd
import numpy as np
import scipy as sp
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image
from numpy import *
%matplotlib inline

arr = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],
               [0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]],dtype= np.int32)
arr.shape
arr.ndim

#im = Image.fromarray(arr)
#im.show()

np.identity(8,int32)
np.zeros((1,3,6))

ar = np.array([1,2,3,4,5])
ar.astype(np.float32)#########convert to float
a = np.arange(10)
a
np.empty(10,dtype= 'O')
np.empty(10,dtype= 'S10')
np.empty(10,dtype= 'U10')
np.empty(10,dtype= '?')

#############
aq =np.array([[1,2,3,4],[1,2,3,4],[4,5,7,8]])
aq*aq
aq-aq
aq+aq
1/aq
aq**2
er = np.array([0,1,2,3,4,5,6,7,8,9])
er_sli = er[5:8].copy()
er_sli
aq[:1,:2]##############slicing

da = np.random.randn(7,4)
da[2:4,:]


q = np.empty((6,8))
for i in arange(6):
    q[i]=i
q
q[[1,3,4,5]]##########select specific rows
q[[-1,-3,-4,-5]]####select in reverse order

aa = np.arange(48).reshape((8,6))
aa[[4,5,6,2],[0,2,5,3]]
aa[[4,5,6,2]][:,[0,2,5,3]]###rctangular selection by fancy index
    
aa[np.ix_([4,5,6,2],[0,2,5,3])]#####same as above selection
aa
aa.T
np.dot(aa,aa.T)
###aa.swapaxes(1,8)
x=np.sqrt(aa)
y= np.exp(aa)
np.maximum(x,y)
x.ndim
aa
np.meshgrid(aa,aa)
z= np.exp(aa**2 + aa**2)
z
#plt.imshow(z,cmap=plt.cm.gray);plt.colorbar()
#plt.title("image plot of ")

#########statistical
h = np.array([[1,2,3],[5,4,7],[9,6,3]])
d =np.random.randn(5,4)
d.mean(), np.mean(d)
d.sum()
d.mean(axis =1)
d.cumsum(1)
h.cumsum(1)
h.argmax(1)##########indices of max
##################################################################np.where() function
x= np.array([1,2,3,4,5])
y = np.array([6,5,4,9,8])
z= np.array([True,False,True,False, False])

#res = [(r if j else a)
#      for r,a,j in zip(x,y,z)]

#res = np.where(z,x,y)
#res




ary = np.random.randn(4,4)
ary
np.where(ary>0,2,-2)


re=[]
np.where(1>2 & 2<1,0,
        np.where(1>2,1,
                np.where(2<1,2,3))) 
list(re)


