import numpy as np
from numba import jit
from numba import types
from .utils import delete_column

float_array = types.float64[:]

@jit(["Tuple((float64[:], float64[:,:], float64[:], float64[:]))\
    (float64[:], float64[:,:], int32, float32)"], fastmath=True,
    parallel=False)
def matching_pursuit(
    signal: np.ndarray,
    dictionary: np.ndarray,
    K: int = 0,
    eps: float = 1e-3
):
    if K <= 0 or K > dictionary.shape[1]:
        K = dictionary.shape[1]
    
    m = len(signal)
    residual = np.ascontiguousarray(signal)
    dictionary = np.ascontiguousarray(dictionary)
    coefficients = np.full(K, 0.)
    atoms = np.full((m, K), 0.)
    errors = np.full(K, 0.)

    k = 0
    while np.sqrt(np.sum(np.square(residual))) > eps:
        dot_product = np.dot(residual, dictionary)
        max_ind = np.argmax(np.abs(dot_product))

        coefficients[k] = dot_product[max_ind]
        atoms[:, k] = dictionary[:, max_ind]

        residual = residual - coefficients[k] * atoms[:, k]
        errors[k] = np.sqrt(np.sum(np.square(residual)))
        # Early stopping when the solution diverges
        if k >= 1 and errors[k] > errors[k-1]:
            print("WARNING: Diverging solution. Ending approximation loop.")
            break
        if abs(errors[k] - errors[k-1]) < eps:
            print("WARNING: Solution not improving. Ending approximation loop.")
            break

        # Removing the deletion step makes computation faster in some cases.
        dictionary = np.ascontiguousarray(delete_column(dictionary, max_ind))

        k += 1
        if k == K or dictionary.size == 0:
            break

    errors = errors[:k]
    coefficients = coefficients[:k]
    atoms = atoms[:, :k]

    return coefficients, atoms, residual, errors

@jit(["Tuple((float64[:], float64[:,:], float64[:], float64[:]))\
    (float64[:], float64[:,:], int32, int32, float32)"], fastmath=True,
    parallel=False)
def orthogonal_matching_pursuit(
    signal: np.ndarray,
    dictionary: np.ndarray,
    K: int = 0,
    N: int = 1,
    eps: float = 1e-3
):
    if K <= 0 or K > dictionary.shape[1]:
        K = dictionary.shape[1]

    m = len(signal)
    signal = np.ascontiguousarray(signal)
    residual = np.ascontiguousarray(signal)
    dictionary = np.ascontiguousarray(dictionary)
    atoms = np.full((m, K), 0., np.float64)
    errors = np.full(K, 0., np.float64)

    k = 0
    while np.sqrt(np.sum(np.square(residual))) > eps  and k < min([K, m//N]):
        dot_product = np.dot(residual, dictionary)
        inds = np.abs(dot_product).argsort()[-N:][::-1]

        atoms[:, k*N:(k+1)*N] = dictionary[:, inds]

        # If dictionary contains complex values,
        # use np.conj(A).T instead of A.T
        A = np.ascontiguousarray(atoms[:, :(k+1)*N])
        estimate = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), signal)
        residual = signal - np.dot(A, estimate)
        errors[k] = np.sqrt(np.sum(np.square(residual)))
        # Early stopping when the solution diverges
        if k >= 1 and errors[k] > errors[k-1]:
            print("WARNING: Diverging solution. Ending approximation loop.")
            break
        if abs(errors[k] - errors[k-1]) < eps:
            print("WARNING: Solution not improving. Ending approximation loop.")
            break

        # Removing the deletion step makes computation faster in some cases.
        dictionary = np.ascontiguousarray(delete_column(dictionary, inds))

        k += 1
        if k == K or dictionary.size == 0:
            break

    errors = errors[:k]
    atoms = np.ascontiguousarray(atoms[:, :k*N])
    coefficients = np.dot(np.dot(np.linalg.inv(np.dot(atoms.T, atoms)),
        atoms.T), signal)
    return coefficients, atoms, residual, errors