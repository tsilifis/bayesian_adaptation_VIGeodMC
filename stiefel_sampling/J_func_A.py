__all__ = ['eval_hermite', 'J', 'J_short']

import numpy as np
import scipy.stats as st 
#import orthpol
import chaos_basispy as cb
from scipy.optimize import minimize


def eval_hermite(xi, alpha):
	assert xi.shape[1] == len(alpha)
	P = np.zeros((xi.shape[0], len(alpha)))
	for i in range(len(alpha)):
		#p = orthpol.OrthogonalPolynomial(alpha[i], rv=st.norm())
		if alpha[i] == 0:
			P[:, i] = np.ones(xi.shape[0])
		else:
			p = cb.Hermite1d(int(alpha[i]))
			P[:, i] = p(xi[:,i])[:,0]
	return np.prod(P, axis = 1)

def J(a, dim, dim_eta, deg, xi, u, coeffs):
	assert a.shape[0] == dim_eta * dim
	assert xi.shape[0] == u.shape[0]
	assert xi.shape[1] == dim
	U = a.reshape((dim_eta, dim))
	rvs = [st.norm()] * dim_eta
	#pol = orthpol.ProductBasis(rvs, degree = deg)
	pol = cb.PolyBasis(dim_eta, deg, 'H')
	Q = pol._MI_terms.shape[0]
	assert coeffs.shape[0] == Q
	eta = np.dot(U, xi.T).T
	Psi_A = pol(eta)
	##### ------- Gradient --------
	Jac = np.zeros(U.shape)
	for i in range(U.shape[0]):
		for j in range(U.shape[1]):
			Psi_grad = np.zeros((xi.shape[0], pol._MI_terms.shape[0]))
			for l in range(1, pol._MI_terms.shape[0]):
				beta = pol._MI_terms[l,:]
				if beta[i] - 1 >= 0:
					alpha = beta - np.eye(dim_eta)[i,:]
					#print beta, alpha
					Psi_grad[:,l] = np.sqrt(beta[i]) * eval_hermite(eta, alpha) * xi[:,j]
			Jac[i,j] = - 2 * np.sum(np.dot(u, Psi_grad) * coeffs) + np.sum(np.dot(coeffs, np.dot(Psi_A.T, Psi_grad) + np.dot(Psi_grad.T, Psi_A)) * coeffs)
#	print Jac
	return np.linalg.norm( u - np.dot(Psi_A, coeffs), 2), Jac.flatten()


def J_short(a, dim, dim_eta, deg, xi, u, coeffs):
	assert a.shape[0] == dim_eta * dim
	assert xi.shape[0] == u.shape[0]
	assert xi.shape[1] == dim
	U = a.reshape((dim_eta, dim))
	rvs = [st.norm()] * dim_eta
	#pol = orthpol.ProductBasis(rvs, degree = deg)
	pol = cb.PolyBasis(dim_eta, deg, 'H')
	Q = pol._MI_terms.shape[0]
	assert coeffs.shape[0] == Q
	eta = np.dot(U, xi.T).T
	Psi_A = pol(eta)
	##### ------- Gradient --------
	Jac = np.zeros(U.shape)
	for i in range(dim_eta-1, U.shape[0]):
		for j in range(U.shape[1]):
			Psi_grad = np.zeros((xi.shape[0], pol._MI_terms.shape[0]))
			for l in range(1, pol._MI_terms.shape[0]):
				beta = pol._MI_terms[l,:]
				if beta[i] - 1 >= 0:
					alpha = beta - np.eye(dim_eta)[i,:]
#					print beta, alpha
					Psi_grad[:,l] = np.sqrt(beta[i]) * eval_hermite(eta, alpha) * xi[:,j]
			Jac[i,j] = - 2 * np.sum(np.dot(u, Psi_grad) * coeffs) + np.sum(np.dot(coeffs, np.dot(Psi_A.T, Psi_grad) + np.dot(Psi_grad.T, Psi_A)) * coeffs)
#	print Jac
	return np.linalg.norm( u - np.dot(Psi_A, coeffs), 2), Jac[-1,:]


