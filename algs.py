import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict

float_array = types.float64[:]

@njit(fastmath=True, parallel=False)
def matching_pursuit(
    signal: np.ndarray,
    dictionary: np.ndarray,
    m: int = -1,
    eps: float = 1e-3
):
    if m == -1:
        m = int(dictionary.shape[1])
        
    residual = np.copy(signal)
    coefficients = np.full((1, m), 0.)
    atoms = np.full((signal.shape[1], m), 0.)
    cache = Dict.empty(
        key_type=types.unicode_type,
        value_type=float_array,
    )
    cache["errors"] = np.full(m, 0., dtype=np.float64)

    i = 0
    while np.sum(np.square(residual)) > eps:
        dot_product = np.dot(residual, dictionary)
        max_ind = np.argmax(np.abs(dot_product))

        coefficients[:, i] = dot_product[:, max_ind]
        atoms[:, i] = dictionary[:, max_ind]

        residual = residual - coefficients[:, i] * atoms[:, i]
        cache["errors"][i] = np.sum(np.square(residual))
        i += 1

        if i == m or dictionary.size == 0:
            break

    coefficients = coefficients[:, :i]
    atoms = atoms[:, :i]

    return coefficients, atoms, residual, cache

@njit(fastmath=True, parallel=False)
def orthogonal_matching_pursuit(
    signal: np.ndarray,
    dictionary: np.ndarray,
    m: int = -1,
    eps: float = 1e-3
):
    if m == -1:
        m = int(dictionary.shape[1])

    residual = np.copy(signal)
    atoms = np.full((signal.shape[1], m), 0.)
    cache = Dict.empty(
        key_type=types.unicode_type,
        value_type=float_array,
    )
    cache["errors"] = np.full(m, 0., dtype=np.float64)

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
        cache["errors"][i] = np.sum(np.square(residual))
        
        i += 1
        if i == m or dictionary.size == 0:
            break
    
    cache["errors"] = cache["errors"][:i]
    atoms = np.ascontiguousarray(atoms[:, :i])
    coefficients = np.dot(signal, np.dot(np.linalg.inv(np.dot(atoms.T, atoms)),
        atoms.T).T)
    return coefficients, atoms, residual, cache

@njit(fastmath=True, parallel=False)
def reconstruct_signal(
    coefficients: np.ndarray,
    atoms: np.ndarray,
    residual: np.ndarray = None,
    signal: np.ndarray = None
):
    reconstructed_signal = np.full((1, len(atoms)), 0.)
    reconstructed_signal[0, :] = np.sum(coefficients * atoms, axis=1)
    if residual is not None:
        reconstructed_signal += residual
    if signal is not None:
        mse = np.sum(np.power(signal - reconstructed_signal, 2))
        rmse = np.sqrt(np.sum(np.power(signal - reconstructed_signal, 2)))
        print("MSE: ", mse)
        print("RMSE:", rmse)
    return reconstructed_signal