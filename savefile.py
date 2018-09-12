import numpy as np
import pandas as pd
import scipy as sp


awe = np.arange(10)
aw = np.arange(48)
np.save('some_array', awe)
np.load('some_array.npy')

##########################  saving multiple arrays

np.savez('arrayarc.npz', a=awe, b=aw)
ax = np.load('arrayarc.npz')
ax['b']
