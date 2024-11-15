""" 

Author: Panos Tsilifis (tsilifis@protonmail.com)
Date: 7/10/2017

"""

import numpy as np 
import scipy.stats as st 
import matplotlib.pyplot as plt 
import sys
sys.path.insert(0, '../../')
from stiefel_sampling import *


# ---- Problem dimensionality -----
d = 10
deg = 2
outpath = 'results/2d/'
# ---- Setting the example: Coeffs, data and chaos model ------
#c = np.array([0.25*12., np.sqrt(12.) + 0.075*(12.**(3./2.)), 0.25*12.*np.sqrt(2.), 0.025*(12.**(3./2)) * np.sqrt(6.)])

XI = np.load('data/XI_2d.npy')[:150,:]
y = np.load('data/y_2d.npy')[:150]#[:,0]

data = {'y': y, 'xi': XI}
params = {'d': 2, 'deg': deg, 'coeffs': np.zeros(6)}
#params_VI = {'omega' : [1, 10.], 'tau': [10., 5.]}
params_VI = {'omega' : [1e-6, 1e-6], 'tau' : [1e-6, 1e-6]}
tau = params_VI['tau'][0] / params_VI['tau'][1]
#tau = 7484.

chaos = ChaosModel(2, deg, np.zeros(6))
#iso1 = np.load(outpath + '/W_2d_1st_iter_4.npy')[:,-1].reshape(d,1)
iso1 = VectorLangevin(d, 0.001).sample(1)[::-1,:]
G = np.linalg.svd(iso1)[0][:,:2]


# ---- Creating likelihood, prior and posterior objects -------
L = Likelihood(data, params, tau)
VL = MatrixLangevin(G, np.array([.001, .001]))
posterior = Posterior(L, VL)

W0 = VL.sample(1)[:,:,0]
#W0 = VectorLangevin(d, 0.01).sample(1)
HMC = GeodesicMC(posterior, eps = 0.005, T = 5)

V0 = VariationalOptimizer(chaos, data, W0, params_VI)

c_chain = np.zeros((6, 1))
w_chain = W0.flatten().reshape(2*d, 1)


c_eta_sol = 1e+4
i = 0
iters = 20

while i < iters and c_eta_sol > 5e-3:
	if i > 2:
		c_eta_sol = np.abs(np.linalg.norm(c_chain[:,i-1]) - np.linalg.norm(c_chain[:,i-2]))
		print '-' * 50
		print '-' * 20 + ' Coefficients relative error ' + '-' * 20
		print '-' * 50
		print c_eta_sol
		print '-' * 50
		print '-' * 50

	#print HMC._target._likelihood._tau
	[c_sol, omega_sol, tau_sol, elbo] = V0.optimize()
	np.save(outpath+'/2d_elbo_vals_2d_iter_'+str(i+1)+'.npy', elbo)
	print '-'*10 + 'Finished Variational Inference' + '-'*10
	print 'Current chaos coefficients solution :'
	print c_sol
	print 'Current scale parameter :'
	print tau_sol
	c_chain = np.hstack([c_chain, c_sol[:,0].reshape(6,1)])
	HMC._target._likelihood._tau = tau_sol[0,0] / tau_sol[0,1]
	HMC._target._likelihood._chaos_model._coeffs = c_sol[:,0].copy()

	chain = HMC.run_chain(10, V0._W)
	print '-'*10+'Finished generating HCM chain' + '-'*10
	print 'Current projection matrix :'
	print chain[-1]
	V0._W = chain[-1].copy()
	V0.update_Psi()
	w_chain = np.hstack([w_chain, chain[-1].flatten().reshape(2*d,1)])

	np.save(outpath + '/c_sol_2d_iter_'+str(i+1)+'.npy', c_sol)
	np.save(outpath + '/W_2d_iter_'+str(i+1)+'.npy', V0._W)

	i = i + 1

if i < iters:
	HMC._target._likelihood._tau = tau_sol[0,0] / tau_sol[0,1]
	HMC._target._likelihood._chaos_model._coeffs = c_sol[:,0].copy()	

	chain = HMC.run_chain(100, V0._W)

	for j in range(len(chain)):
		w_chain = np.hstack([w_chain, chain[j].flatten().reshape(2*d,1)])
	print 'Early termination of coefficients update.'
	print 'Run an additional chain for W'


np.save(outpath +'/iso_1d_data_ord2_2d.npy', w_chain)
np.save(outpath + '/c_1d_data_ord2_2d.npy', c_chain)



