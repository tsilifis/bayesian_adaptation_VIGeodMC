"""
author: Panos Tsilifis (tsilifis@protonmail.com)
date: 11/14/2024
"""

import numpy as np 
import scipy.stats as st 
import matplotlib.pyplot as plt 
import sys
sys.path.insert(0, '../../')
from stiefel_sampling import MatrixLangevin

dim = 10
np.random.seed(12345)
W = np.random.normal(size = (dim,2))
W = W.reshape(W.shape[0],2) / np.linalg.norm(W)
W0 = MatrixLangevin(W, np.array([0.001,0.001])).sample(1)[:,:,0]

print 'W : '
print W0
np.save('data/W_2d_true.npy', W0)
a = np.random.normal()
b = np.random.normal(size = (2,1))
c = np.random.normal(size = (2,2))
print 'a : '
print a
print 'b : '
print b
print 'c : '
print c

xi = st.norm.rvs(size = (10,1000))
np.save('data/XI_2d.npy', xi.T)
#xi = st.norm.rvs(size = (10, 10000))
def f(xi, a, b, c, W):
    assert xi.shape[0] == 10
    assert W.shape[0] == 10
    assert b.shape[0] == 2
    assert c.shape[0] == 2
    return a + np.dot(b.T, np.dot(W.T, xi) ) + np.dot(np.dot(xi.reshape(1,xi.shape[0]), W), np.dot(np.dot(c, W.T) , xi))

out = np.zeros(1000)
for i in range(1000):
	out[i] = f(xi[:,i], a ,b ,c, W0)
#def grad_f(xi, ):
#    pass
np.save('data/y_2d.npy', out)

plt.style.use('ggplot')

plt.hist(out, bins =100, normed = True)
plt.show()


#sparse = np.zeros(grad_f.shape)
#sparse[:5,:5] = grad_f_1
#sparse[5:,5:] = grad_f_2
#[l0, v0] = np.linalg.eigh(sparse)
#print 'eigenvalues & eigenvectors \n'
#print l, v
#print l1, v1
#print l2, v2

#out = np.zeros(1000)
#for i in range(1000):
#    out[i] = f(xi[:,i], a, b, c, W)

#z = np.dot(W.T, xi)

#eta = np.dot(W.T, st.uniform.rvs(loc = -1., scale = 2., size = (dim, 100000)))
