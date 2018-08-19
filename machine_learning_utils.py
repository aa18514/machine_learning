import numpy as np
from sklearn.decomposition import PCA

def standardize(data, mean, std, epsilon=10**-8):
    """standardize to zero mean and unit variance"""
    return (data - mean)/(std + epsilon)


def pca_transformation(train_data, test_data, n_features):
     pca = PCA(n_components=n_features)
     pca.fit(train_data)
     modified_train_features = np.ones((train_data.shape[0], n_features+1))
     modified_test_features = np.ones((test_data.shape[0], n_features+1))
     modified_train_features[:, 1:] = pca.transform(train_data)
     modified_test_features[:, 1:]  = pca.transform(test_data)
     return modified_train_features, modified_test_features
