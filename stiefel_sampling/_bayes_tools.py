__all__ = ['ChaosModel', 'Likelihood', 'Posterior', 'Metropolis', 'GeodesicMC']

import numpy as np 
import scipy.stats as st 
import scipy.linalg as lng
import chaos_basispy as cb
from J_func_A import *


class ChaosModel(object):

	_dim = None

	_order = None

	_basis = None

	_coeffs = None

	def __init__(self, dim, order, coeffs):
		"""
		Initializes the object
		"""
		assert isinstance(dim,int)
		assert isinstance(order,int)
		self._dim = dim
		self._order = order
		self._basis = cb.PolyBasis(self._dim, self._order, 'H')
		#print self._basis._MI_terms.shape[0]
		assert self._basis._MI_terms.shape[0] == coeffs.shape[0]
		self._coeffs = coeffs

	def eval(self, xi):
		return np.dot(self._basis(xi), self._coeffs)

class SparseChaosModel(object):

	_dim = None

	_order = None

	_basis = None

	_coeffs = None

	_sparse_z = None

	def __init__(self, dim, order, coeffs, z):
		"""
		Initializes the object
		"""
		assert isinstance(dim, int)
		assert isinstance(order, int)
		self._dim = dim
		self._order = order 
		self._basis = cb.PolyBasis(self._dim, self._order, 'H')
		assert self._basis._MI_terms.shape[0] == coeffs.shape[0]
		assert self._basis._MI_terms.shape[0] == z
		self._coeffs = coeffs
		self._sparse_z = z

	def eval(self, xi):
		C = self._coeffs * self._sparse_z
		return np.dot(self._basis(xi), C)


class Likelihood(object):


	_data = None

	_K = None

	_inp_dim = None

	_chaos_model = None

	_tau = None

	def __init__(self, data, chaos_params, scale):
		"""
		Initializes the object
		"""

		assert data['y'].shape[0] == data['xi'].shape[0]
		self._data = data
		self._K = data['y'].shape[0]
		self._inp_dim = data['xi'].shape[1]
		assert chaos_params['d'] <= self._inp_dim
		self._chaos_model = ChaosModel(chaos_params['d'], chaos_params['deg'], chaos_params['coeffs'])
		assert scale > 0
		self._tau = scale

	def eval(self, W):
		assert W.shape[0] == self._chaos_model._dim
		assert W.shape[1] == self._inp_dim

		eta = np.dot(W, self._data['xi'].T).T
		#print eta
		PsiC = self._chaos_model.eval(eta)
		logL = - (self._K / 2.) * np.log( 2*np.pi / self._tau) - (self._tau / 2.) * np.linalg.norm(self._data['y'] - PsiC) ** 2
		return logL

	def eval_L(self, W):
		assert W.shape[0] == self._chaos_model._dim
		assert W.shape[1] == self._inp_dim

		eta = np.dot(W, self._data['xi'].T).T
		#print eta
		PsiC = self._chaos_model.eval(eta)
		L = - (self._tau / 2.) * np.linalg.norm(self._data['y'] - PsiC) ** 2
		return L	

	def eval_grad_W(self, W, fixcols = False):
		if not fixcols:
			[J0, DJ] = J(W.flatten(), self._inp_dim, self._chaos_model._dim, self._chaos_model._order, self._data['xi'], self._data['y'], self._chaos_model._coeffs)
			return - self._tau * DJ.reshape(self._chaos_model._dim, self._inp_dim) / 2.
		else:
			[J0, DJ] = J_short(W.flatten(), self._inp_dim, self._chaos_model._dim, self._chaos_model._order, self._data['xi'], self._data['y'], self._chaos_model._coeffs)
			return - self._tau * DJ.reshape(1,self._inp_dim) / 2.

class Posterior(object):

	_likelihood = None

	_prior = None

	def __init__(self, likelihood, prior):
		"""
		Initializes the object
		"""
		self._likelihood = likelihood
		self._prior = prior

	def eval_logp(self, W):
		return self._likelihood.eval(W) + self._prior.eval_logp(W)

	def eval_grad_logp(self, W, complement = None, fixcols = False):
		if not fixcols:
			return (self._likelihood.eval_grad_W(W) + self._prior.eval_grad_logp(W)).T
		else: 
			grad = np.vstack([complement.T, self._likelihood.eval_grad_W(W, fixcols)])
			return (grad + self._prior.eval_grad_logp(W)).T


class Metropolis(object):

	_proposal = None

	_like = None

	_chain = None

	def __init__(self, prop, like):
		self._proposal = prop
		self._like = like

	def acceptance(self, W):
		return np.min([1., np.exp(self._like.eval(W) - self._like.eval(self._chain[-1])) ])

	def run_chain(self, N = 1000):
		self._chain = [self._proposal.sample(1).T]
		i = 1
		while i < N:
			W = self._proposal.sample(1).T
			if st.uniform.rvs() < self.acceptance(W):
				print self.acceptance(W) 
				print W
				self._chain += [W]
				i = i + 1
				print 'Chain current length : ' + str(len(self._chain))


class GeodesicMC(object):

	_target = None 

	_eps = None

	_T = None

	def __init__(self, target, eps = 0.01, T = 5):
		"""
		Initializes the object
		"""
		assert isinstance(T, int)
		self._target = target
		self._eps = eps
		self._T = T

	def run_chain(self, M, W0):
		X = W0
		chain = [X.copy()]
		n = 0 
		d = W0.shape[0]
		p = W0.shape[1]
		if p > 1:
			post_grad = self._target.eval_grad_logp(X.T)[:,:p-1].reshape(d,p-1)
			while n < M:
				u = st.norm.rvs(size = (d, p))
				u_proj = u - np.dot(X, np.dot(X.T, u) + np.dot(u.T, X)) / 2.

				H = self._target.eval_logp(X.T) - np.linalg.norm(u_proj.flatten()) / 2.
				print 'Hamiltonian :' + ' '*10 + str(H)
				x_star = X.copy()
				for i in range(self._T):
					u_new = u_proj + self._eps * self._target.eval_grad_logp(x_star.T, complement = post_grad, fixcols = True) / 2.
					u_new_proj = u_new - np.dot(x_star, np.dot(x_star.T, u_new) + np.dot(u_new.T, x_star)) / 2.
					#u_new_proj = u_new - np.dot(X, np.dot(X.T, u_new) + np.dot(u_new.T, X)) / 2.
					A = np.dot(x_star.T, u_new_proj)
					S_0 = np.dot(u_new_proj.T, u_new_proj)
					V_0 = np.hstack([x_star, u_new_proj])
					exp1 = np.vstack([ np.hstack([A, -S_0]), np.hstack([np.eye(X.shape[1]), A]) ])
					exp2 = np.vstack([ np.hstack([ lng.expm(-self._eps*A), np.zeros((p,p)) ]), np.hstack([np.zeros((p,p)), lng.expm(-self._eps*A)]) ])
					#print 'exp2 : ' + str(exp2)
					V_eps = np.dot(V_0, np.dot(lng.expm(self._eps * exp1), exp2))
					x_star = V_eps[:,:p].reshape(d,p).copy()
					u_new = V_eps[:,p:].reshape(d,p).copy()
					u_new = u_new + self._eps * self._target.eval_grad_logp(x_star.T, complement = post_grad, fixcols = True) / 2.
					#print 'u_new : ' + str(u_new)
					u_new_proj = u_new - np.dot(x_star, np.dot(x_star.T, u_new) + np.dot(u_new.T, x_star) ) / 2.
					#u_new_proj = u_new - np.dot(X, np.dot(X.T, u_new) + np.dot(u_new.T, X)) / 2.

				H_new = self._target.eval_logp(x_star.T) - np.linalg.norm(u_new_proj.flatten()) / 2.
				print 'Proposed Hamiltonian :' + str(H_new)
				if st.uniform.rvs() < np.exp(H_new - H):
					X = x_star.copy()
					chain += [X]
					n = n + 1
					print n
		elif p == 1:
			#chain = X.copy()
			while n < M:
				u = st.norm.rvs(size = (d,1))
				u_proj = np.dot((np.eye(d) - np.dot(X, X.T)), u)

				H = self._target.eval_logp(X.T) - np.linalg.norm(u_proj) / 2.
				print 'Hamiltonian' +' '*10 + ':' + str(H)
				x_star = X.copy()
				for i in range(self._T):
					u_new = u_proj + self._eps * self._target.eval_grad_logp(x_star.T) / 2.
					u_new_proj = np.dot(np.eye(d) - np.dot(x_star, x_star.T), u_new)
					#u_new_proj = np.dot(np.eye(d) - np.dot(X, X.T), u_new)

					alpha = np.linalg.norm(u_new_proj)
					V_0 = np.hstack([x_star, u_new_proj])
					a = np.array([1., alpha])
					rot = np.array([ [np.cos(alpha*self._eps), -np.sin(alpha*self._eps)], [np.sin(alpha*self._eps), np.cos(alpha*self._eps)] ])
					V_eps = np.dot(V_0, np.dot(np.diag(1./a), np.dot(rot, np.diag(a)) ))
					x_star = V_eps[:,0].reshape(d,1).copy()
					u_new = V_eps[:,1].reshape(d,1).copy()
					u_new = u_new + self._eps * self._target.eval_grad_logp(x_star.T) / 2.
					u_new_proj = np.dot(np.eye(d) - np.dot(x_star, x_star.T), u_new)
					#u_new_proj = np.dot(np.eye(d) - np.dot(X, X.T), u_new)
					u_proj = u_new_proj.copy()

				H_new = self._target.eval_logp(x_star.T) - np.linalg.norm(u_new_proj) / 2.
				print 'Proposed Hamiltonian :' + str(H_new)
				if st.uniform.rvs() < np.exp(H_new - H):
					X = x_star.copy()
					chain += [X]
					n = n + 1
					print n

		return chain







"""
c = np.array([0.25*12., np.sqrt(12.) + 0.075*(12.**(3./2.)), 0.25*12.*np.sqrt(2.), 0.025*(12.**(3./2)) * np.sqrt(6.)])
#chaos = ChaosModel(2, 4, c)
#xi = st.norm.rvs(size = (1000,2))
XI = np.load('toy/XI_toy.npy')
y = np.load('toy/q_toy.npy')

data = {'y': y, 'xi': XI}
params = {'d': 1, 'deg': 3, 'coeffs': c}
tau = 0.5

chaos = ChaosModel(1, 3, c)

L = Likelihood(data, params, tau)

W = np.ones((1,12)) / np.sqrt(12.)

eta = np.dot(W, XI.T).T
print L.eval(W)

import matplotlib.pyplot as plt 

plt.plot(chaos.eval(eta), 'o')
plt.plot(y, '.')
plt.show()

from stiefel_sampling import *

VL = VectorLangevin(12, 0.01)
mh_sampler = Metropolis(VL, L)

mh_sampler.run_chain(1000)
chain = mh_sampler._chain
np.save('chain.npy', chain)

"""
