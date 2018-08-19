import numpy as npi
from sklearn.decomposition import PCA
from typing import List
from scipy.stats as stats
import matplotlib.axes.Axes

Vector_float = List[float]

def standardize(data: Vector_float, mean: float, std: float, epsilon: float=10**-8)->(Vector_float, float, float):
    """standardize to zero mean and unit variance"""
    return (data - mean)/(std + epsilon)


def pca_transformation(train_data: Vector_float, test_data: Vector_float, n_features:int)->(Vector_float, Vector_float, int):
     pca = PCA(n_components=n_features)
     pca.fit(train_data)
     modified_train_features = np.ones((train_data.shape[0], n_features+1))
     modified_test_features = np.ones((test_data.shape[0], n_features+1))
     modified_train_features[:, 1:] = pca.transform(train_data)
     modified_test_features[:, 1:]  = pca.transform(test_data)
     return modified_train_features, modified_test_features

def exponential_weighted_average(error: Vector_float, beta=0.9)->(Vector_float):
     """for details of how EMEA is calculated please refer to
     https://en.wikipedia.org/wiki/Moving_average"""
     vo = 0.0
     error = (1 - beta) * error
     vs = []
     for i in range(len(error)):
         vo = (1 - beta) * vo + beta*error[i]
         vs.append(vo)
     return vs

def get_normalized_residuals(residuals: Vector_float)->Vector_float:
     centered_residual = (residuals - np.mean(residuals))**2
     weighted_residual = centered_residual/np.sum(centered_residual)
     normalized_residual = residuals/(np.var(residuals) * (1 - weighted_residual - (1/len(residuals))))
     return normalized_residual

 def histogram_residuals(residuals: Vector_float, ax: Axes):
     weights = np.ones_like(residuals)/float(len(residuals))
     fit = stats.norm.pdf(np.sort(residuals), np.mean(residuals), np.std(residuals))
     ax.hist(residuals, weights=weights, bins=75, color='r')
     ax.plot(np.sort(residuals), fit, c='k', linestyle='-')
     ax.text(5, 0.22, r'$\mu=%f,\ \sigma=%f$' % (np.mean(fit), np.std(fit)))

