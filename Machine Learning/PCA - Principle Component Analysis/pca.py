import numpy as np

class PCA:
    def __init__(self, method='cov'):
        """
            method - 'cov': use covariance method
                   - 'svd': use singular value decomposition method
            WARNING: the SVD method currently is experiencing a poor performance!
        """
        if method not in ['cov', 'svd']:
            raise Exception('invalid method name. method name should be either \'cov\' or \'svd\'')
        self.method = method
        self.N = 0 # number of samples
        self.M = 0 # original dimension
        self.mu = None # mean of each dimension
        self.cov = None # covariance matrix
        self.pc = None # principle components

    def fit(self, x, k):
        """
            x - data: NxM matrix (N samples, M dimensions)
            k - new dimension
        """
        x_ = np.copy(x)
        self.N = x_.shape[0] # Number of samples
        self.M = x_.shape[1] # Original dimension
        if self.M < k:
            raise Exception('new dimension should be less than or equal to original dimension, but got: '+str(self.N)+'=>'+str(k))
        self.mu = np.mean(x_, axis=0) # Compute mean for each dimension
        x_ = x_ - self.mu # Subtract off the mean for each dimension
        if self.method == 'cov':
            """ Compute the covariance matrix """
            self.cov = 1 / (self.N-1) * np.matmul(np.transpose(x_), x_)
            """ Find the (sorted already) eigenvalues and eigenvectors (principle components) """
            (v, self.pc) = np.linalg.eig(self.cov)
            """ Project the original dataset """
            x_ = np.matmul(x_, self.pc[:,:k]) 
        else:
            """ Construct the matrix y_ """
            y_ = x_ / np.sqrt(self.N-1)
            """ Singular Value Decomposition (SVD) """
            (u, s, self.pc) = np.linalg.svd(y_)
            """ Project the original dataset """
            x_ = np.matmul(x_, self.pc[:,:k])
        return x_

    def transform(self, x, k):
        """
            x - data: NxM matrix (N samples, M dimensions)
            k - new dimension
        """
        x_ = np.copy(x)
        x_ = x_ - self.mu # Subtract off the mean for each dimension
        if self.M != x_.shape[1]:
            raise Exception('dimension not compatible with the fitted model')
        if self.M < k:
            raise Exception('new dimension should be less than or equal to original dimension, but got: '+str(self.N)+'=>'+str(k))
        """ Project the original dataset """
        x_ = np.matmul(x_, self.pc[:,:k])
        return x_

