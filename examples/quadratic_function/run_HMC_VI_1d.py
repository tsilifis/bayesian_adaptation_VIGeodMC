""" 

Author: Panos Tsilifis (tsilifis@protonmail.com)
Date: 7/10/2017

"""

import numpy as np 
import sys
sys.path.insert(0, '../../')
from stiefel_sampling import *
import scipy.stats as st 
import matplotlib.pyplot as plt 
np.random.seed(1234)

# ---- Problem dimensionality -----
d = 10
deg = 2
datapath = 'data'
outpath = 'results/1d'
# ---- Setting the example: Coeffs, data and chaos model ------
#c = np.array([0.25*12., np.sqrt(12.) + 0.075*(12.**(3./2.)), 0.25*12.*np.sqrt(2.), 0.025*(12.**(3./2)) * np.sqrt(6.)])

XI = np.load(datapath+'/XI_1d.npy')[:150,:]
y = np.load(datapath+'/y_1d.npy')[:150]

data = {'y': y, 'xi': XI}
params = {'d': 1, 'deg': deg, 'coeffs': np.zeros(deg+1)}
params_VI = {'omega' : [1e-6, 1e-6], 'tau': [1e-6, 1e-6]}
tau = params_VI['tau'][0] / params_VI['tau'][1]
#tau = 7484.

chaos = ChaosModel(1, deg, np.zeros(deg+1))

# ---- Creating likelihood, prior and posterior objects -------
L = Likelihood(data, params, tau)
VL = MatrixLangevin(np.ones((d,1)) / np.sqrt(d), np.array([0.001]))
posterior = Posterior(L, VL)

W0 = VectorLangevin(d, 0.001).sample(1)[::-1,:]
HMC = GeodesicMC(posterior, eps = 0.005, T = 10)

V0 = VariationalOptimizer(chaos, data, W0, params_VI)

iters = 40
c_chain = np.zeros((deg+1, 1))
w_chain = W0
#omega_chain = np.zeros((5, iters))
tau_chain = np.zeros((2, iters))

#def update(VI, HMC, update = 0):
#	assert update in [-1, 0, 1]
#	if update == 0:
#		[c_sol, omega_sol, tau_sol] = VI.optimize()

c_eta_tol = 1e+4
i = 0
while i < iters and c_eta_tol > 1e-3:
	if i > 2:
		c_eta_tol = np.abs(np.linalg.norm(c_chain[:,i-1]) - np.linalg.norm(c_chain[:,i-2]))
		print '-' * 50
		print '-' * 20 + ' Coefficients relative error ' + '-' * 20
		print '-' * 50
		print c_eta_tol

	#print HMC._target._likelihood._tau
	[c_sol, omega_sol, tau_sol, elbo] = V0.optimize()
	np.save(outpath+'/1d_elbo_vals_eps005_T10_iter_'+str(i+1)+'.npy', elbo)
	print '-'*10 + 'Finished Variational Inference' + '-'*10
	print 'Current chaos coefficients solution :'
	print c_sol
	print 'Current scale parameter :'
	print tau_sol
	c_chain = np.hstack([c_chain, c_sol[:,0].reshape(deg+1,1)])
	#np.save('scram_results/c_sol_iter_'+str(i+1)+'.npy', c_sol)
	tau_chain[:,i] = tau_sol.flatten()
	#np.save('scram_results/tau_sol_iter_'+str(i+1)+'.npy', tau_sol)
	HMC._target._likelihood._tau = tau_sol[0,0] / tau_sol[0,1]
	HMC._target._likelihood._chaos_model._coeffs = c_sol[:,0].copy()	

	chain = HMC.run_chain(10, V0._W)	
	print '-'*10+'Finished generating HCM chain' + '-'*10
	print 'Current projection matrix :'
	print chain[-1]
	V0._W = chain[-1].copy()
	w_chain = np.hstack([w_chain, chain[-1]])
	#np.save(outpath+'/w_sol_iter_'+str(i+1)+'.npy', chain[-1])
	V0.update_Psi()

	np.save(outpath+'/c_sol_1d_iter_'+str(i+1)+'.npy', c_sol)
	np.save(outpath+'/W_1d_iter_'+str(i+1)+'.npy', V0._W)

	i = i + 1

if i < iters:
	HMC._target._likelihood._tau = tau_sol[0,0] / tau_sol[0,1]
	HMC._target._likelihood._chaos_model._coeffs = c_sol[:,0].copy()	

	chain = HMC.run_chain(500, V0._W)

	for j in range(len(chain)):
		w_chain = np.hstack([w_chain, chain[j]])
	print 'Early termination of coefficients update.'
	print 'Run an additional chain for W'


#np.save(path+'/2d_results/c_2nd_ord2_1d.npy', c_chain)
#np.save(path+'/2d_results/iso_2nd_ord2_1d.npy', w_chain)
#np.save(path+'/2d_results/tau_2nd_ord2_1d.npy', tau_chain)



