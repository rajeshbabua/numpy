############ linear algebra

from numpy.linalg import inv, qr
x = np.random.randn(5,5)
(x.T.dot(x)).dot(inv(x.T.dot(x)))


sam = np.random.randn(4,4)
sam = np.random.normal(size=(4,4))
sam = np.random.seed()
sam
