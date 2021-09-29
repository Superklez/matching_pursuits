import numpy as np

def matching_pursuit(signal, dictionary, m=None, eps=1e-3):
    if m is None:
        m = dictionary.shape[1]
        
    residual = np.copy(signal)
    coefficients = np.zeros((1, m))
    atoms = np.zeros((signal.shape[1], m))
    cache = {"errors":[]}

    i = 0
    while np.sum(np.square(residual)) > eps:
        dot_product = np.dot(residual, dictionary)
        max_ind = np.argmax(np.abs(dot_product))

        coefficients[:, i] = dot_product[:, max_ind]
        atoms[:, i] = dictionary[:, max_ind]

        residual = residual - coefficients[:, i] * atoms[:, i]
        cache["errors"].append(np.sum(np.square(residual)))
        dictionary = np.delete(dictionary, max_ind, axis=1)
        i += 1

        if i == m or dictionary.size == 0:
            break

    coefficients = coefficients[:, :i]
    atoms = atoms[:, :i]

    return coefficients, atoms, residual, cache

def orthogonal_matching_pursuit(signal, dictionary, m=None, eps=1e-3):
    if m is None:
        m = dictionary.shape[1]
    
    residual = np.copy(signal)
    atoms = np.zeros((signal.shape[1], m))
    cache = {"errors":[]}

    i = 0
    while np.sum(np.square(residual)) > eps:
        dot_product = np.dot(residual, dictionary)
        max_ind = np.argmax(np.abs(dot_product))

        atoms[:, i] = dictionary[:, max_ind]
        i += 1

        P = np.matmul(atoms[:, :i], np.matmul(np.linalg.inv(np.matmul(
            np.transpose(np.conj(atoms[:, :i])), atoms[:, :i])),
            np.transpose(np.conj(atoms[:, :i]))))
        residual = signal - np.dot(signal, P)
        cache["errors"].append(np.sum(np.square(residual)))
        dictionary = np.delete(dictionary, max_ind, axis=1)

        if i == m or dictionary.size == 0:
            break

    atoms = atoms[:, :i]
    coefficients = np.dot(signal, np.matmul(np.linalg.inv(np.matmul(
        np.transpose(np.conj(atoms)), atoms)), np.transpose(np.conj(atoms))).T)
    
    return coefficients, atoms, residual, cache
    
def reconstruct_signal(coefficients, atoms, residual=None, signal=None):
    reconstructed_signal = np.sum(coefficients * atoms, axis=1,
        keepdims=True).T
    if residual is not None:
        reconstructed_signal += residual
    if signal is not None:
        mse = np.sum(np.power(signal - reconstructed_signal, 2))
        rmse = np.sqrt(mse)
        print(f"MSE:  {mse}")
        print(f"RMSE: {rmse}")