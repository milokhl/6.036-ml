import random
import time

import numpy as np
from math import log, isnan
from scipy.stats import multivariate_normal
from numpy.linalg import norm

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
        # get useful dimensions
        n, D = data.shape # num example, num features
        ds = np.shape(self.params['alpha'][0])[1] # does return an int
        k = self.k

        # calculate the posteriors for each x_i
        posteriorArray = np.zeros((n, k))
        for i in range(n):
            x_i = np.array(data.iloc[i]) # get x_i for convenience
            numSum = 0 # stores sum of numerator

            for j in range(k): # for each cluster
                num = self.params['pi'][j]
                for d in range(D):
                    if isnan(x_i[d]):
                        num *= 1
                    else:
                        num *= self.params['alpha'][d][j][int(x_i[d])]
                numSum += num
                posteriorArray[i][j] = num

            # normalize to get the actual posteriors
            posteriorArray[i] /= numSum
        
        # compute log-likelihood
        ll = 0
        for i in range(n):
            x_i = np.array(data.iloc[i]) # get x_i for convenience
            for j in range(k): # for each cluster
                ll += posteriorArray[i][j] * log(self.params['pi'][j])
                for d in range(D): # for each feature
                    if isnan(x_i[d]):
                        ll += 0 # because [[x_d^i is missing]]
                    else:
                        ll += posteriorArray[i][j] * log(self.params['alpha'][d][j][int(x_i[d])])

        return (ll, posteriorArray)

    def m_step(self, data, p_z):
        """ Performs the M-step of the EM algorithm
        data - an NxD pandas DataFrame
        p_z - an NxK numpy ndarray containing posterior probabilities

        returns a dictionary containing the new parameter values
        """
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

        # compute new alpha
        for d in range(D):
            n_d = self.params['alpha'][d].shape[1] # determine number of possible values this d could take on
            for j in range(k): # for each cluster
                for cat in range(n_d):
                    hasValCtr = 0
                    alphaSum = 0
                    for i in range(n):
                        if isnan(data.iloc[i, d]):
                            pass
                        else:
                            alphaSum += p_z[i][j] * (int(data.iloc[i, d]) == cat)
                            hasValCtr += p_z[i][j]
                    new_alpha[d][j][cat] = float(alphaSum) / hasValCtr

        return {
            'pi': new_pi,
            'alpha': new_alpha,
        }

    @property
    def bic(self):
        raise NotImplementedError()
