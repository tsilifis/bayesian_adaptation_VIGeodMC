"""
author: Panos Tsilifis (tsilifis@protonmail.com)
date: 11/14/2024
"""

import numpy as np 
import scipy.stats as st 
import matplotlib.pyplot as plt 
#import chaospy as cp
import chaos_basispy as cb
#import sys 
#sys.path.insert(0,'../adaptation_tools/')
#import tools


def f(xi, a, b, c, W):
    assert xi.shape[0] == 10
    assert W.shape[0] == 10
    return a + b * np.dot(W.T, xi) + c * np.dot(xi.reshape(1,xi.shape[0]), np.dot(np.dot(W, W.T) , xi))


dim = 10
np.random.seed(12345)
W = np.random.normal(size = (dim,))
W = W.reshape(W.shape[0],1) / np.linalg.norm(W)
print 'W : ' 
print W
np.save('data/W_1d_true.npy', W)
a = np.random.normal()
b = np.random.normal()
c = np.random.normal()
print ' a ' + ' '*10+'|'+ ' b ' + ' '*10 +'|'+ ' c '
print a,b,c

xi = st.norm.rvs(size = (10,1000))
np.save('data/XI_1d.npy', xi.T)

out = np.zeros(1000)
for i in range(1000):
    out[i] = f(xi[:,i], a, b, c, W)
    #out[i] += out[i] * 0.1 * st.norm.rvs()
np.save('data/y_1d.npy', out)


z = np.dot(W.T, xi)

eta = np.dot(W.flatten(), st.uniform.rvs(loc = -1., scale = 2., size = (dim, 100000)))

plt.style.use('ggplot')

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.hist(out, bins = 100, normed = True, histtype = 'stepfilled')
ax1.set_xlabel('Value', fontsize = 15)
ax1.set_ylabel('pdf', fontsize = 15)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)  
ax1.get_xaxis().tick_bottom()  
ax1.get_yaxis().tick_left() 
for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_fontsize(15)
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(15)
#plt.legend(prop={'size':28})
#plt.savefig('fig1.png')
plt.show()


hist, bins = np.histogram(eta, bins=500)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()

#distr1 = cp.J(cp.Uniform(-1,1))
#zeta, weights = cp.generate_quadrature(20, distr1, "C")
#zeta = zeta.T

