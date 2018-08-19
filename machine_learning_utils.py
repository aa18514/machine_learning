
def standardize(data, mean, std, epsilon=10**-8):
    """standardize to zero mean and unit variance"""
    return (data - mean)/(std + epsilon)
