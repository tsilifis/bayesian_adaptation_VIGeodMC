__all__ = ['ExponFam', 'VariationalOptimizer']

import numpy as np 
import scipy.stats as st 
import scipy.special as sp


class ExponFam(object):
	"""
	"""
	_pdf = None

	def __init__(self, pdf):
		"""
		Initializes the object
		"""
		self._pdf = pdf

	def A(self, eta):
		if self._pdf == 'Gauss':
			return - eta[0]**2/(4*eta[1]) - np.log(-2*eta[1]) / 2.
		elif self._pdf == 'Gamma':
			return np.log( sp.gamma(eta[0]+1) ) - (eta[0]+1.) * np.log(-eta[1])


	def A_grad(self, eta):
		if self._pdf == 'Gauss':
			return np.array([-eta[0]/(2*eta[1]), eta[0]**2 / (4*eta[1]**2) - 1./(2*eta[1])])
		elif self._pdf == 'Gamma':
			return np.array([sp.digamma(eta[0]+1) - np.log(-eta[1]) , -(eta[0]+1)/eta[1]])

	def map_from_eta(self, eta):
		if self._pdf == 'Gauss':
			return np.array([-eta[0]/(2*eta[1]), -2*eta[1]])
		elif self._pdf == 'Gamma':
			return np.array([eta[0]+1., -eta[1]])

	def map_to_eta(self, params):
		if self._pdf == 'Gauss':
			return np.array([params[0]*params[1], -params[1] / 2.])
		elif self._pdf == 'Gamma':
			return np.array([params[0]-1., -params[1]])


class VariationalOptimizer(object):

	_chaos_model = None

	_data = None

	_W = None

	_Psi = None

	_prior_params = None
	# Number of observations 
	_K = None
	# L component
	_expL = None

	def __init__(self, chaos, data, W, prior_params):
		"""
		Initializes the object
		"""
		assert chaos._dim == W.shape[1]
		self._chaos_model = chaos
		self._data = data
		self._W = W
		self._Psi = self._chaos_model._basis(np.dot(W.T, self._data['xi'].T).T)
		self._prior_params = prior_params
		self._K = data['xi'].shape[0]

	def z_c_mean(self, omega):
		exp_F = ExponFam('Gamma')
		eta = exp_F.map_to_eta(omega)
		return np.array([0, (eta[0]+1) / (2*eta[1])])

	def z_tau(self):
		exp_F = ExponFam('Gamma')
		return exp_F.map_to_eta(self._prior_params['tau'])

	def z_omega(self):
		exp_F = ExponFam('Gamma')
		return exp_F.map_to_eta(self._prior_params['omega'])

	def update_Psi(self):
		self._Psi = self._chaos_model._basis(np.dot(self._W.T, self._data['xi'].T).T)

	def update_L(self):
		eta = np.dot(self._W.T, self._data['xi'].T).T
		self._L = np.array([self._K / 2., - np.linalg.norm(self._data['y'] - self._chaos_model.eval(eta) ) ** 2 / 2.])

	def expL(self, m, rho):
		eta = np.dot(self._W.T, self._data['xi'].T).T
		Psi = self._chaos_model._basis(eta)
		self._expL = np.array([self._K / 2., - np.linalg.norm(self._data['y'] - np.dot(Psi, m) ) ** 2 / 2. - np.sum(np.diag(np.dot(np.dot(Psi.T, Psi), np.diag(1/rho) )  / 2.) ) ])

	def compELBO(self, eta_c, eta_om, eta_tau):
		expF_gauss = ExponFam('Gauss')
		expF_gamma = ExponFam('Gamma')
		ELBO = - self._K * np.log(2*np.pi) / 2. + np.sum(expF_gamma.A_grad(eta_tau) * self._expL)
		for i in range(eta_c.shape[0]):
			ELBO += expF_gamma.A(eta_om[i,:]) - np.sum( (eta_om[i,:] - self.z_omega())* expF_gamma.A_grad(eta_om[i,:]) ) - expF_gamma.A(self.z_omega())
			omega = expF_gamma.map_from_eta(eta_om[i,:])
			ELBO += expF_gauss.A(eta_c[i,:]) - np.sum( eta_c[i,:] * expF_gauss.A_grad(eta_c[i,:]) ) - np.sum( self.z_c_mean(omega) * expF_gauss.A_grad(eta_c[i,:]) ) - (-sp.digamma(eta_om[i,0]+1) + np.log(-eta_om[i,0]) ) / 2.
		ELBO +=  - (eta_tau[0]+1.) * np.log(-eta_tau[1]) + (self.z_tau()[0]+1.) * np.log(-self.z_tau()[1]) - np.sum( eta_tau - self.z_tau() * np.array([sp.digamma(eta_tau[0]+1) - np.log(-eta_tau[1]) , -(eta_tau[0]+1)/eta_tau[1]]) )
		#print 'ELBO : ' + str(ELBO)
		return ELBO


	def optimize(self):
		expF_gauss = ExponFam('Gauss')
		expF_gamma = ExponFam('Gamma')
		eta_c = np.zeros((self._chaos_model._coeffs.shape[0],2))
		eta_om = np.zeros((self._chaos_model._coeffs.shape[0],2))
		for i in range(self._chaos_model._coeffs.shape[0]):
			eta_om[i,:] = self.z_omega()
			omega = expF_gamma.map_from_eta(eta_om[i,:])
			eta_c[i,:] = self.z_c_mean(omega)
		eta_tau = self.z_tau().reshape((1,2))
		elbo = np.array([-np.inf])

		eta_c_new = 1e+5 * np.ones(eta_c.shape)
		eta_om_new = 1e+5 * np.ones(eta_om.shape)
		eta_tau_new = 1e+5 * np.ones(eta_tau.shape)

		c_sol = np.zeros(eta_c.shape)
		omega_sol = np.zeros(eta_om.shape)
		tau_sol = np.zeros(eta_tau.shape)
		#print 'Eta_c :' + str(eta_c)
		err = 1e+6
		iters = 0
		while err > 1e-4:
			params_c = np.zeros(eta_c.shape)
			for i in range(params_c.shape[0]):
				params_c[i,:] = expF_gauss.map_from_eta(eta_c[i,:])
			self.expL(params_c[:,0], params_c[:,1])
			eta_tau_new = self.z_tau() + self._expL# - np.array([20., 0])
			print eta_tau_new
			print 'Updating tau'
			for i in range(eta_om.shape[0]):
				eta_om_new[i,:] = self.z_omega() + np.array([1., - expF_gauss.A_grad(eta_c[i,:])[1]]) / 2.
			print 'Updating omega'
			for i in range(eta_c.shape[0]):
				e_i = np.zeros(eta_c.shape[0])
				e_i[i] = 1.
				v_i = np.array([np.dot(self._data['y'].reshape(1, self._K), np.dot(self._Psi, e_i)), - np.sum(np.diag(np.dot(self._Psi.T, np.dot(self._Psi, np.diag(e_i))))) / 2. ])
				m_i = params_c[:,0].copy().reshape(eta_c.shape[0], 1)
				m_i[i,0] = 0.
				em_i = np.dot(e_i.reshape(eta_c.shape[0],1), m_i.T)
				u_i = np.array([- np.sum(np.diag( np.dot(self._Psi.T, np.dot(self._Psi, em_i)) )), 0])
				eta_c_new[i,:] = (v_i + u_i) * expF_gamma.A_grad(eta_tau_new)[1] + self.z_c_mean(eta_om_new[i,:])
				params_c[i,:] = expF_gauss.map_from_eta(eta_c_new[i,:])
			print 'Updating c'
			ETA = np.vstack([eta_tau, eta_om, eta_c])
			ETA_new = np.vstack([eta_tau_new, eta_om_new, eta_c_new])
			err = np.linalg.norm(ETA-ETA_new)
			print 'Relative error : ' + str(err)
			elbo = np.append(elbo, self.compELBO(eta_c_new, eta_om_new, eta_tau_new) )
			eta_c = eta_c_new.copy()
			eta_tau = eta_tau_new.copy()
			eta_om = eta_om_new.copy()
			iters += 1

		tau_sol[:] = expF_gamma.map_from_eta(eta_tau)
		for i in range(c_sol.shape[0]):
			c_sol[i,:] = expF_gauss.map_from_eta(eta_c[i,:])
			omega_sol[i,:] = expF_gamma.map_from_eta(eta_om[i,:])
			
		#print tau_sol
		#print c_sol
		#print omega_sol
		print 'Total number of iterations : ' + str(iters)
		return c_sol, omega_sol, tau_sol, elbo

