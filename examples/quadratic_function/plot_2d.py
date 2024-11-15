""" 

Author: Panos Tsilifis (tsilifis@protonmail.com)
Date: November 14, 2024

"""

import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import chaos_basispy as cb
import scipy.stats as st 

plt.style.use('ggplot')


c = np.load('results/2d/c_sol_2d_iter_13.npy')[:,0]
iso = np.load('results/2d/W_2d_iter_13.npy')

print iso.shape

pol = cb.PolyBasis(1, 2)
XI = np.load('data/XI_2d.npy')[150:250,:]
q = np.load('data/y_2d.npy')[150:250]
W = np.load('data/W_2d_true.npy')


pol2 = cb.PolyBasis(2, 2)

eta = np.dot(iso.T, XI.T).T

x_mesh = np.linspace(-4., 4., 50)
y_mesh = np.linspace(-4., 4., 50)
X, Y = np.meshgrid(x_mesh,y_mesh)

pol = cb.PolyBasis(2, 2)
Z = np.zeros(X.shape)
for i in range(X.shape[0]):
	for j in range(X.shape[1]):
		xi = np.array([X[i,j], Y[i,j]]).reshape(1,2)
		Z[i,j] = np.dot(pol(xi), c)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(eta[:,0], eta[:,1], q, 'r.', s = 1.5)

#ax.scatter(eta[:,0], eta[:,1], y_2d, 'r.', s = 1.)

ax.plot_surface(X, Y, Z, rstride = 8, cstride = 8, alpha = 0.3)
ax.view_init(20,200)
ax.set_xlabel(r'$\eta_1$', fontsize = 12)
ax.set_ylabel(r'$\eta_2$', fontsize = 12)
ax.set_zlabel(r'$f(\eta_1, \eta_2)$', fontsize = 12)
plt.tight_layout()
#plt.savefig('images/f_2d_full.pdf')

plt.show()


#print iso_all.shape

fig_W = plt.figure()
ax_1 = fig_W.add_subplot(211)
ax_1.plot(range(1, 11), iso[:, 0], linestyle = '', color = 'C1', marker = 'x', ms = 10.)
ax_1.plot(range(1, 11), W[:,0], linestyle = '', marker = 'o')
ax_1.set_xlabel(r'$W_{:,1}$', fontsize = 12)
ax_1.set_ylabel('Value', fontsize = 12)
ax_2 = fig_W.add_subplot(212)
ax_2.plot(range(1, 11), iso[:, 1], linestyle = '', color = 'C1', marker = 'x', ms = 10.)
ax_2.plot(range(1, 11), W[:,1], linestyle = '', marker = 'o')
ax_2.set_xlabel(r'$W_{:,2}$', fontsize = 12)
ax_2.set_ylabel('Value', fontsize = 12)
plt.tight_layout()
#plt.savefig('images/2d_entries.eps')
plt.show()