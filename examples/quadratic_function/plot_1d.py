import numpy as np 
import matplotlib.pyplot as plt 

import chaos_basispy as cb
import scipy.stats as st 

plt.style.use('ggplot')


c = np.load('results/2d/c_sol_2d_1st_iter_4.npy')[:,0]
iso = np.load('results/2d/iso_1st_ord2_2d.npy')[:, -1]
#print np.load('2d_results/W_1d_iter_20.npy')

print iso

pol = cb.PolyBasis(1, 2)
XI = np.load('data/XI_2d.npy')[:150,:]
q = np.load('data/y_2d.npy')[:150]
W = np.load('data/W_2d_true.npy')[:, 1]

#error = [-W.flatten() - np.min(iso[:,6:], axis = 1), np.max(iso[:,6:], axis = 1) + W.flatten()]
#print error[0]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(1,11), W, color = 'C7', linestyle = '', marker = 's', ms = 5, label = True)
ax.plot(range(1,11), iso, color='C1', linestyle = '', marker = 'x', ms = 10, label='Posterior sample')
#ax.errorbar(range(1, 11), -W, yerr = error[0], fmt = ' ', ms = 15, ecolor = (0.4,0.0,1.0) , color = 'C1', label = 'Posterior bars')
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(12)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(12)
ax.set_xlabel('W entry', fontsize = 12)
ax.set_ylabel('Value', fontsize = 12)
plt.legend(fontsize = 12)
plt.tight_layout()
#plt.savefig('images/iso_1d.eps')
plt.show()


pol = cb.PolyBasis(1, 2)
x = np.linspace(-3.5, 3.5, 501) 

fig_c = plt.figure()
ax = fig_c.add_subplot(111)
ax.plot(x, np.dot(pol(x.reshape(501,1)), c), label = 'f')
ax.plot(np.dot(iso[:,-1], XI.T), q, linestyle = ' ', marker = 's', ms = 3., label = 'Data')
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(12)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(12)
ax.set_xlabel(r'$\eta$', fontsize = 12)
ax.set_ylabel(r'$f(\eta)$', fontsize = 12)
plt.legend(fontsize = 12)
plt.tight_layout()
#plt.savefig('images/f_1d.eps')
plt.show()


fig_s = plt.figure()
ax1 = fig_s.add_subplot(111)
ax1.plot(range(1, 507), - np.ones((506, 10)) * W.flatten(), color = 'k', linewidth = 1.5)
ax1.plot(range(1, 507), iso.T, color = 'C1', linewidth = 1.)
#plt.legend(fontsize = 12)
ax1.set_xlabel('Iteration', fontsize = 12)
ax1.set_ylabel('Value', fontsize = 12)
plt.tight_layout()
#plt.savefig('images/trace_1d.eps')
plt.show()

