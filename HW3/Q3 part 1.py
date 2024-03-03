import numpy as np

def lift(input_vector):
    dim = len(input_vector)
    # Include original input_vector in the lifted vector
    lifted = [input_vector[i] for i in range(dim)]
    # Add products of coordinates
    for i in range(dim):
        for j in range(i + 1):  
            lifted.append(input_vector[i] * input_vector[j])

    x_lifted = np.array(lifted)
    return x_lifted


def liftDataset(X):
    n, d = X.shape
    # Apply lift to each row of X
    X_lifted = np.array([lift(X[i, :]) for i in range(n)])
    return X_lifted
