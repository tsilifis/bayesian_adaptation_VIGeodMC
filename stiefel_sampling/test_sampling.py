import numpy as np
import scipy.stats as st 
import scipy.special as sp
from _stiefel_sampling import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('seaborn-notebook')

#X = uniform(5,4, 10000)

W = VectorLangevin(3, 5000.)
X = W.sample(1000)

#plt.hist(X[0,:], bins = 100, normed = True)
#plt.show()

u = st.norm.rvs(size = (3,))
u = u / np.linalg.norm(u)
print u
Y = map_vL(X,u)

#print np.linalg.norm(X, axis = 0)

#plt.style.use('seaborn')

fig = plt.figure(figsize = (10,4))
ax1 = fig.add_subplot(121, projection = '3d')
ax1.scatter(X[0,:], X[1,:], X[2,:], s = 1)
ax1.plot([0,0], [0,0], [0,1])
ax2 = fig.add_subplot(122, projection = '3d')
ax2.scatter(Y[0,:], Y[1,:], Y[2,:], s = 1)
ax2.plot([0,u[0]], [0,u[1]], [0, u[2]] )
plt.show()


U = st.norm.rvs(size = (3,3))
u0, s0, v0 = np.linalg.svd(U)
print u0.shape
s0[0] = 5000.
s0[1] = .01
mL = MatrixLangevin(u0[:,:2], s0[:2])
Z = mL.sample(1000)
print Z[0,0,:]

#plt.hist(Z[0,0,:], bins = 100, normed = True)
#plt.show()

fig_mL = plt.figure(figsize = (5,5))
ax = fig_mL.add_subplot(111, projection = '3d')
ax.scatter(Z[0,0,:], Z[1,0,:], Z[2,0,:], s = 1)
ax.plot([0, u0[0,0]], [0, u0[1,0]], [0, u0[2,0]])
ax.scatter(Z[0,1,:], Z[1,1,:], Z[2,1,:], s = 1)
ax.plot([0, u0[0,1]], [0, u0[1,1]], [0, u0[2,1]])
plt.show()