import numpy as np
from numba import jit
from numba import types
from utils import delete_column

float_array = types.float64[:]

@jit(["Tuple((float64[:,:], float64[:,:], float64[:,:], float64[:]))\
    (float64[:,:], float64[:,:], int32, float32)"], fastmath=True,
    parallel=False)
def matching_pursuit(
    signal: np.ndarray,
    dictionary: np.ndarray,
    m: int = 0,
    eps: float = 1e-3
):
    if m <= 0 or m > dictionary.shape[1]:
        m = dictionary.shape[1]
        
    residual = np.ascontiguousarray(np.copy(signal))
    dictionary = np.ascontiguousarray(dictionary)
    coefficients = np.full((1, m), 0.)
    atoms = np.full((signal.shape[1], m), 0.)
    errors = np.full(m, 0.)

    i = 0
    while np.sum(np.square(residual)) > eps:
        dot_product = np.dot(residual, dictionary)
        max_ind = np.argmax(np.abs(dot_product))

        coefficients[:, i] = dot_product[:, max_ind]
        atoms[:, i] = dictionary[:, max_ind]

        residual = residual - coefficients[:, i] * atoms[:, i]
        errors[i] = np.sum(np.square(residual))

        # Removing the deletion step makes computation faster in some cases.
        dictionary = np.ascontiguousarray(delete_column(dictionary, max_ind))

        i += 1
        if i == m or dictionary.size == 0:
            break

    errors = errors[:i]
    coefficients = coefficients[:, :i]
    atoms = atoms[:, :i]

    return coefficients, atoms, residual, errors

@jit(["Tuple((float64[:,:], float64[:,:], float64[:,:], float64[:]))\
    (float64[:,:], float64[:,:], int32, float32)"], fastmath=True,
    parallel=False)
def orthogonal_matching_pursuit(
    signal: np.ndarray,
    dictionary: np.ndarray,
    m: int = 0,
    eps: float = 1e-3
):
    if m <= 0 or m > dictionary.shape[1]:
        m = dictionary.shape[1]

    signal = np.ascontiguousarray(signal)
    residual = np.ascontiguousarray(signal)
    dictionary = np.ascontiguousarray(dictionary)
    atoms = np.full((signal.shape[1], m), 0., np.float64)
    errors = np.full(m, 0., np.float64)

    i = 0
    while np.sum(np.square(residual)) > eps:
        dot_product = np.dot(residual, dictionary)
        max_ind = np.argmax(np.abs(dot_product))

        atoms[:, i] = dictionary[:, max_ind]

        # If dictionary contains complex values,
        # use np.conj(A).T instead of A.T
        A = np.ascontiguousarray(atoms[:, :i+1])
        P = np.dot(A, np.dot(np.linalg.inv(np.dot(A.T, A)), A.T))
        residual = signal - np.dot(signal, P)
        errors[i] = np.sum(np.square(residual))

        # Removing the deletion step makes computation faster in some cases.
        dictionary = np.ascontiguousarray(delete_column(dictionary, max_ind))
        
        i += 1
        if i == m or dictionary.size == 0:
            break
    
    errors = errors[:i]
    atoms = np.ascontiguousarray(atoms[:, :i])
    coefficients = np.dot(signal, np.dot(np.linalg.inv(np.dot(atoms.T, atoms)),
        atoms.T).T)
    return coefficients, atoms, residual, errors