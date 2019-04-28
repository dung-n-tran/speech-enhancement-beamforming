import numpy as np
class MicrophoneArray(object):
    def __init__(self, array_geometry):
        self.dim = array_geometry.shape[0]
        self.n_mics = array_geometry.shape[1]
        self.geometry = array_geometry

class BaseDLBeamformer(object):
    def __init__(self, vs):
        """
        Parameters
        ----------
        vs: Source manifold array vector
        """
        self.vs = vs
        self.weights_ = None
        
    def _compute_weights(self, training_data):
        n_training_samples = len(training_data)
        n_mics, snapshot = training_data[0].shape
        D = np.zeros((n_mics, n_training_samples), dtype=complex)
        for i_training_sample in range(n_training_samples):
            nv = training_data[i_training_sample]
            Rnhat = nv.dot(nv.transpose().conjugate()) / snapshot
            Rnhatinv = np.linalg.inv(Rnhat)
            w = Rnhatinv.dot(self.vs) / (self.vs.transpose().conjugate().dot(Rnhatinv).dot(self.vs))
            D[:, i_training_sample] = w.reshape(n_mics,)
        return D

    def _initialize(self, X):
        pass

    def _choose_weights(self, x):
        n_dictionary_atoms = self.weights_.shape[1]
        R = x.dot(x.transpose().conjugate())
        proxy = np.diagonal(self.weights_.transpose().conjugate().dot(R).dot(self.weights_))
        optimal_weight_index = np.argmin(proxy)

#         min_energy = np.inf
#         optimal_weight_index = None
#         for i_dictionary_atom in range(n_dictionary_atoms):
#             w = self.weights_[:, i_dictionary_atom]
#             energy = np.real(w.transpose().conjugate().dot(R).dot(w))
#             if min_energy > energy:
#                 min_energy = energy
#                 optimal_weight_index = i_dictionary_atom
        return self.weights_[:, optimal_weight_index]
    
    def fit(self, training_data):
        """
        Parameters
        ----------
        X: shape = [n_samples, n_features]
        """
        D = self._compute_weights(training_data)
        self.weights_ = D
        return self

    def choose_weights(self, x):
        return self._choose_weights(x)