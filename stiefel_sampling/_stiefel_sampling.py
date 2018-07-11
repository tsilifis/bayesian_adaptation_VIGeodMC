
__all__ = ['Uniform', 'VectorLangevin', 'map_vL', 'MatrixLangevin']


import numpy as np 
import scipy.stats as st 
import scipy.special as sp

class Uniform(object):

	_dim = None

	_frame = None

	"""docstring for  Uniform"""
	def __init__(self, frame, dim):
		assert isinstance(frame, int)
		assert isinstance(dim, int)
		self._frame = frame
		self._dim = dim
		
	def sample(self, N = 1):
		assert isinstance(n, int)
		assert isinstance(d, int)
		assert isinstance(N, int)
		Z = st.norm.rvs(size = (n, d, N))
		X = np.zeros((n, d, N))
		for i in range(N):
			Z_inv = np.linalg.inv(np.dot(Z[:,:,i].T,Z[:,:,i]))
		#Z_inv = np.linalg.inv(np.dot(Z.T, Z))
			[l,v] = np.linalg.eigh(Z_inv)
			V = v[:,::-1]
			S = np.sqrt(np.diag(l[::-1]))
			X[:,:,i] = np.dot(Z[:,:,i], np.dot(V, np.dot(S, V.T)) )
		return X

class VectorLangevin(object):

	""" Dimensionality """
	_dim = None
	""" Parameter """
	_kappa = None
	""" Normalizing constant """
	_C = None

	"""docstring for W"""
	def __init__(self, dim, kappa):
		self._dim = dim
		self._kappa = kappa
		self._C = np.sqrt(np.pi) * (self._kappa / 2.)**((self._dim - 2.)/2.) * sp.iv((self._dim - 2.) / 2., self._kappa) * sp.gamma((self._dim - 1.) / 2.)

	def sample(self, N = 100):
		b = ( -2.*self._kappa + np.sqrt(4.*self._kappa**2 + (self._dim - 1.)**2)) / (self._dim - 1.)
		x0 = (1.-b)/(1.+b)
		c = self._kappa*x0 + (self._dim - 1.)*np.log(1.-x0**2)
		W = np.zeros((self._dim, N))
		i = 0
		while i < N:
			Z = st.beta((self._dim - 1.)/2., (self._dim - 1.)/2.).rvs()
			U = st.uniform.rvs()
			w0 = (1. - (1.+b)*Z) / (1. - (1.-b)*Z)
			if self._kappa*w0 + (self._dim - 1.)*np.log(1. - x0*w0) - c >= np.log(U):
				V = st.norm.rvs(size = (self._dim-1,))
				V = V / np.linalg.norm(V)
				W[:,i] = np.hstack([np.sqrt(1 - w0**2) * V, w0])
				i = i + 1
		return W

def map_vL(X, w):
	"""
	Maps a random sample drawn from vector Langevin with orientation u = [0,...,0,1] to 
	a sample that follows vector Langevin with orientation w.
	"""
	assert w.shape[0] == X.shape[0]
	#assert np.linalg.norm(w) == 1.
	print 'Orientation vector length : ' + str(np.linalg.norm(w))
	d = w.shape[0]
	w = w.reshape(w.shape[0],1)
	H = np.eye(d) - 2 * np.dot(w, w.T)
	[l, v] = np.linalg.eigh(H)
	V = v[:,::-1]
	if np.sum( w.flatten()*V[:,-1] ) < 0:
		V[:,-1] = -V[:,-1].copy()
	return np.dot(V, X)


class MatrixLangevin(object):

	""" Dimension """
	_dim = None
	""" Orientation parameter """
	_G = None
	""" Scaling parameter """
	_k = None

	def __init__(self, G, k):
		assert G.shape[0] >= G.shape[1]
		assert G.shape[1] == k.shape[0]
		self._G = G
		self._k = k
		self._dim = G.shape[0]

	def D(self):
		D = 1.
		for i in range(self._G.shape[1]):
			D *= sp.gamma((self._dim - i) / 2.) * sp.iv((self._dim - i - 2.)/2., self._k[i]) / (self._k[i] / 2.)**((self._dim - i -2.)/2.)
		return D

	def sample(self, N = 100):
		vL = VectorLangevin(self._dim, self._k[0])
		W = map_vL(vL.sample(N), self._G[:,0])
		X = np.zeros((self._G.shape[0], self._G.shape[1], N))
		i = 0
		while i < N:
			X[:,0,i] = W[:,i]
			D_N = sp.gamma((self._dim)/2.) * sp.iv((self._dim - 2.)/2., np.linalg.norm(self._k[0] * self._G[:,0])) / np.linalg.norm( self._k[0] * self._G[:,0] / 2.) ** ((self._dim-2.)/2.)
			for j in range(1,self._G.shape[1]):
				w = X[:,:j,i].reshape(W.shape[0], j)
				U,S,V = np.linalg.svd(w)
				Null = U[:,j:]
				new_orient = np.dot(Null.T, self._G[:,j])
				new_orient = new_orient/ np.linalg.norm(new_orient)
				#print new_orient.shape
				z = map_vL(VectorLangevin(new_orient.shape[0], self._k[j]).sample(1), new_orient)
				
				x_unsc = np.dot(Null, z)
				#print x_unsc
				x = x_unsc / np.linalg.norm(x_unsc)
				#print np.linalg.norm(x)
				#print np.sum(w * x)
				#X[:,j,i] = x.flatten()
				D_N *= sp.gamma((self._dim - j)/2.) * sp.iv((self._dim - j-2.)/2., np.linalg.norm(self._k[j] * new_orient)) / np.linalg.norm( self._k[j] * new_orient / 2.) ** ((self._dim-j-2.)/2.)
			#print D_N / self.D()
			#if st.uniform.rvs() <= D_N / self.D():
			X[:,j,i] = x.flatten()
			i = i + 1
			print i
		return X

	def eval_logp(self, W):
		F = np.dot(self._G, np.diag(self._k))
		return np.trace(np.dot(F.T,W.T))

	def eval_grad_logp(self, W):
		return np.dot(self._G, np.diag(self._k)).T

