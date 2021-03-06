import random
import time

import numpy as np
from math import log, isnan
from scipy.stats import multivariate_normal
from numpy.linalg import norm
import pandas as pd

def k_means(data, k, eps=1e-4, mu=None):
    """ Run the k-means algorithm
    data - an NxD pandas DataFrame
    k - number of clusters to fit
    eps - stopping criterion tolerance
    mu - an optional KxD ndarray containing initial centroids

    returns: a tuple containing
        mu - a KxD ndarray containing the learned means
        cluster_assignments - an N-vector of each point's cluster index
    """
    if type(data) != type(np.array(0)): # if data is not a numpy array
        data = np.array(data) # convert it to one

    n, d = data.shape
    if mu is None:
        # randomly choose k points as initial centroids
        mu = data[random.sample(range(data.shape[0]), k)]

    # stores cluster index assigned to each of n points
    assignments = np.zeros(n)
    cluster_sizes = np.zeros(k)
    cluster_sums = np.zeros((k,d))

    # store the previous mu to detect when to stop
    prev_mu = np.zeros((k,d))

    converged = False
    iters = 0
    while (not converged):
        iters += 1
        print("Iter:", iters)

        # assign each point to the euclidean closest cluster
        for pt in range(n):
            cluster_id = 0
            best_euc_dist = float('inf')

            for u in range(k):
                resid = data[pt]-mu[u]
                euc_dist = np.linalg.norm(resid, ord=2)

                if euc_dist < best_euc_dist:
                    cluster_id = u
                    best_euc_dist = euc_dist

            assignments[pt] = cluster_id
            cluster_sizes[cluster_id] += 1
            cluster_sums[cluster_id] += data[pt]

        # recompute means: divide each cluster sum by the num pts. in that cluster
        for i in range(k):
            mu[i] = cluster_sums[i] / cluster_sizes[i]

        # check if converged
        for i in range(k):
            if np.linalg.norm((mu[i] - prev_mu[i]), ord=2) > eps: # if a mean has changed by more than epsilon
                converged = False
            else:
                converged = True

        # make sure this is here!
        prev_mu = mu
        
    return (mu, assignments)


class MixtureModel(object):
    def __init__(self, k):
        self.k = k
        self.params = {
            'pi': np.random.dirichlet([1]*k),
        }

    def __getattr__(self, attr):
        if attr not in self.params:
            raise AttributeError()
        return self.params[attr]

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def e_step(self, data):
        """ Performs the E-step of the EM algorithm
        data - an NxD pandas DataFrame

        returns a tuple containing
            (float) the expected log-likelihood
            (NxK ndarray) the posterior probability of the latent variables
        """
        raise NotImplementedError()

    def m_step(self, data, p_z):
        """ Performs the M-step of the EM algorithm
        data - an NxD pandas DataFrame
        p_z - an NxK numpy ndarray containing posterior probabilities

        returns a dictionary containing the new parameter values
        """
        raise NotImplementedError()

    @property
    def bic(self):
        """
        Computes the Bayesian Information Criterion for the trained model.
        Note: `n_train` and `max_ll` set during @see{fit} may be useful
        """
        raise NotImplementedError()

    def fit(self, data, eps=1e-4, verbose=True, max_iters=100):
        """ Fits the model to data
        data - an NxD pandas DataFrame
        eps - the tolerance for the stopping criterion
        verbose - whether to print ll every iter
        max_iters - maximum number of iterations before giving up

        returns a boolean indicating whether fitting succeeded

        if fit was successful, sets the following properties on the Model object:
          n_train - the number of data points provided
          max_ll - the maximized log-likelihood
        """
        last_ll = np.finfo(float).min
        start_t = last_t = time.time()
        i = 0
        while True:
            i += 1
            if i > max_iters:
                return False
            ll, p_z = self.e_step(data)
            print("LL Score:", ll)
            new_params = self.m_step(data, p_z)
            self.params.update(new_params)
            if verbose:
                dt = time.time() - last_t
                last_t += dt
                print('iter %s: ll = %.5f  (%.2f s)' % (i, ll, dt))
                last_ts = time.time()
            if abs((ll - last_ll) / ll) < eps:
                break
            last_ll = ll

        setattr(self, 'n_train', len(data))
        setattr(self, 'max_ll', ll)
        self.params.update({'p_z': p_z})

        print('max ll = %.5f  (%.2f min, %d iters)' %
              (ll, (time.time() - start_t) / 60, i))

        #print('bic: %f' % self.bic)

        return True


class GMM(MixtureModel):
    def __init__(self, k, d):
        super(GMM, self).__init__(k)
        self.params['mu'] = np.random.randn(k, d)

    def e_step(self, data):
        """ Performs the E-step of the EM algorithm
        data - an NxD pandas DataFrame

        returns a tuple containing
            (float) the expected log-likelihood
            (NxK ndarray) the posterior probability of the latent variables
        """
        if type(data) != type(np.array(0)): # if data is not a numpy array
            print("[INFO] Converting pandas DataFrame to numpy array.")
            data = np.array(data) # convert it to one

        n, d = data.shape
        k = self.k
        posteriorArray = np.zeros((n, k)) # stores all p(j | i)

        # set up all the gaussians
        gaussians = []
        for j in range(k):
            norm_j = multivariate_normal(mean=self.params['mu'][j], cov=self.params['sigsq'][j])
            gaussians.append(norm_j)

        # for each point, compute the posterior
        posteriorSums = np.zeros(n)
        for i in range(n):
            for j in range(k):
                posterior = self.params['pi'][j] * gaussians[j].pdf(data[i]) # compute numerator of posterior
                posteriorArray[i][j] = posterior
                posteriorSums[i] += posterior

        # compute log likelihoods before normalizing
        logLikelihood = np.sum(np.log(np.sum(posteriorArray, 1)))

        # divide every element in posteriorArray by the corresponding sum in posteriorSums
        for i in range(n):
            posteriorArray[i] /= posteriorSums[i]

        return (logLikelihood, posteriorArray)


    def m_step(self, data, pz_x):
        """ Performs the M-step of the EM algorithm
        data - an NxD pandas DataFrame
        p_z - an NxK numpy ndarray containing posterior probabilities

        returns a dictionary containing the new parameter values
        """
        n, d = data.shape
        k = self.k

        # store the expected number of points that each gaussian should claim
        expNumPointsEachGaussian = sum(pz_x, 0) # sum down column
        new_pi = expNumPointsEachGaussian / n

        new_mu = np.zeros((k,d))
        for j in range(k):
            mu_sum = 0
            for i in range(n):
                mu_sum += pz_x[i][j] * data[i]
            new_mu[j] = mu_sum / expNumPointsEachGaussian[j]

        new_sigsq = np.zeros(k)
        for j in range(k):
            sig_sum = 0
            for i in range(n):
                sig_sum += (pz_x[i][j] * norm((data[i]-new_mu[j]), ord=2)**2)
            new_sigsq[j] = sig_sum / (2.0 * expNumPointsEachGaussian[j])

        return {
            'pi': new_pi,
            'mu': new_mu,
            'sigsq': new_sigsq,
        }

    def fit(self, data, *args, **kwargs):
        self.params['sigsq'] = np.asarray([np.mean(data.var(0))] * self.k)
        return super(GMM, self).fit(data, *args, **kwargs)


class CMM(MixtureModel):
    def __init__(self, k, ds):
        """d is a list containing the number of categories for each feature"""
        super(CMM, self).__init__(k)
        # each alpha is a (k x d) numpy ndarray
        # each cluster k has a length n_d array of probabilities that sum to one
        self.params['alpha'] = [np.random.dirichlet([1]*d, size=k) for d in ds] # ds = [4,2,3] in test example

    def e_step(self, data):
        print(" *** E STEP *** ")
        n, D = data.shape # num example, num features
        ds = np.shape(self.params['alpha'][0])[1] # does return an int
        k = self.k

        posteriorArray = np.zeros((n, k))
        unfilled = True
        for d in range(D):
            x_d = data.iloc[:,d]
            dummy = pd.get_dummies(x_d)
            res1 = np.dot(dummy, self.params['alpha'][d].T)
            res1[res1 == 0] = 1 # replace all zeros with ones (to handle NaN values)
            if unfilled:
                posteriorArray = res1
                unfilled = False
            else:
                posteriorArray = np.multiply(posteriorArray, res1)

        for i in range(n):
            posteriorArray[i] = np.multiply(posteriorArray[i], self.params['pi']) # multiply by corresponding pi
            posteriorArray[i] /= np.sum(posteriorArray[i]) # normalize

        ll = np.sum(np.dot(posteriorArray, np.log(self.params['pi']))) # add the first part of the ll formula
        for d in range(D):
            x_d = data.iloc[:,d]
            dummy = pd.get_dummies(x_d)
            res2 = np.dot(dummy, self.params['alpha'][d].T)
            res2[res2 == 0] = 1 # replace all zeroes with ones (to handle NaN values)
            ll += np.sum(np.multiply(np.log(res2), posteriorArray)) # ptwise multiply and sum

        return (ll, posteriorArray)

    def m_step(self, data, p_z):
        print(" *** M STEP *** ")
        # get useful dimensions
        n, D = data.shape # num example, num features
        ds = np.shape(self.params['alpha'][0])[1] # does return an int
        k = self.k
        new_pi = np.array(k)
        new_alpha = self.params['alpha']

        # calculate each n_j
        expNumPointsEachCluster = np.sum(p_z, 0) # sum down columns

        # compute new pi by normalizing
        new_pi = np.divide(expNumPointsEachCluster, n)

        for d in range(D): # for each item in alpha
            x_d = data.iloc[:,d]
            dummy = pd.get_dummies(x_d)
            new_alpha[d] = np.dot(p_z.T, dummy)
            for j in range(k):
                new_alpha[d][j] /= np.sum(new_alpha[d][j])

        return {
            'pi': new_pi,
            'alpha': new_alpha,
            }

    def getBIC(self):
        D = len(self.params['alpha'])
        num_params = 0
        num_params += (self.k - 1) # pi params

        # alpha params
        for d in range(D):
            num_params += self.params['alpha'][d].shape[0] * (self.params['alpha'][d].shape[1]-1)

        bic = self.max_ll - 0.5 * log(self.n_train) * num_params
        return bic

    @property
    def bic(self):
        """
        BIC(D, theta) = ll(D;theta) - 0.5 * dim(theta) * log(n)
        """
        D = len(self.params['alpha'])
        num_params = 0
        num_params += (self.k - 1) # pi params

        # alpha params
        for d in range(D):
            num_params += self.params['alpha'][d].shape[0] * (self.params['alpha'][d].shape[1]-1)

        bic = self.max_ll - 0.5 * log(self.n_train) * num_params
        return bic




