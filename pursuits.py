import numpy as np
from numba import jit
from numba import types
from .utils import delete_column

float_array = types.float64[:]

@jit(["Tuple((float64[:], float64[:,:], float64[:], int32[:]))\
    (float64[:], float64[:,:], int32, float32)",
    "Tuple((float32[:], float32[:,:], float32[:], int32[:]))\
    (float32[:], float32[:,:], int32, float32)"], fastmath=True,
    parallel=False)
def matching_pursuit(
    signal: np.ndarray,
    dictionary: np.ndarray,
    K: int = 0,
    maxerr: float = 0.05
):
    """
    Applies the matching pursuit algorithm.
    Inputs:
    --------
    signal (float 1D): Signal to perform sparse approximation via matching
        pursuit on.
    dictionary (float 2D): Dictionary of atoms to base the expansion on.
    K (int): Maximum sparsity desired; maximum number of atoms to extract.
    maxerr (float): Maximum relative error in calculating the ratio of the
        L2-norms of the residual and the original signal.

    Outputs:
    --------
    coefficients (float 1D): Matrix of coefficients.
    atoms (float 2D): Matrix of atoms extracted.
    residual (float 1D): Residual after termination main loop.
    indices (float 1D): Column indices of the selected atoms.
    """

    # Limit the maximum sparsity to the number of atoms, i.e., all atoms may
    # be used.
    if K <= 0 or K > dictionary.shape[1]:
        K = dictionary.shape[1]
    
    # Store the residual and dictionary as contiguous arrays in memory to speed
    # up the calculation of the dot product.
    residual = np.ascontiguousarray(signal)
    dictionary = np.ascontiguousarray(dictionary)

    # Initialize coefficients, atoms, and errors
    m = len(signal)
    coefficients = np.full(K, 0., dtype=signal.dtype)
    atoms = np.full((m, K), 0., dtype=signal.dtype)
    indices = np.full(K, 0., dtype=np.int32)
    all_inds = np.arange(dictionary.shape[1], dtype=np.int32)

    # The original signal's L2-norm is constant, so we calculate it before of
    # the loop to avoid calculating it for each iteration.
    signal_l2 = np.sqrt(np.sum(np.square(signal)))

    # Begin main loop.
    k = 0
    while  np.sqrt(np.sum(np.square(residual))) / signal_l2 > maxerr:
        # Determine which atom is most correlated to the current residual and
        # determine its index.
        dot_product = np.dot(residual, dictionary)
        max_ind = np.argmax(np.abs(dot_product))
        indices[k] = all_inds[max_ind]

        # Assign the maximum dot product (in magnitude) to the kth order
        # coefficient, and the dictionary that yielded the maximum dot product
        # to the kth order atom.
        coefficients[k] = dot_product[max_ind]
        atoms[:, k] = dictionary[:, max_ind]

        # Update residual.
        residual = residual - coefficients[k] * atoms[:, k]

        # Remove selected atoms from dictionary.
        dictionary = np.ascontiguousarray(delete_column(dictionary, max_ind))
        all_inds = np.delete(all_inds, max_ind)

        k += 1
        # If we have reached the desired sparsity or there are no atoms left
        # in the dictionary, then terminate the main loop.
        if k == K or dictionary.size == 0:
            break

    indices = indices[:k]
    coefficients = coefficients[:k]
    atoms = atoms[:, :k]

    return coefficients, atoms, residual, indices

@jit(["Tuple((float64[:], float64[:,:], float64[:], int32[:]))\
    (float64[:], float64[:,:], int32, int32, float32)",
    "Tuple((float32[:], float32[:,:], float32[:], int32[:]))\
    (float32[:], float32[:,:], int32, int32, float32)"], fastmath=True,
    parallel=False)
def orthogonal_matching_pursuit(
    signal: np.ndarray,
    dictionary: np.ndarray,
    K: int = 0,
    N: int = 1,
    maxerr: float = 0.05
):
    """
    Applies the generalized orthogonal pursuit algorithm.
    Inputs:
    --------
    signal (float 1D): Signal to perform sparse approximation via matching
        pursuit on.
    dictionary (float 2D): Dictionary of atoms to base the expansion on.
    K (int): Maximum sparsity desired; maximum number of atoms to extract.
    N (int): Number of atoms to extract in each iteration.
    maxerr (float): Maximum relative error in calculating the ratio of the
        L2-norms of the residual and the original signal.

    Outputs:
    --------
    coefficients (float 1D): Matrix of coefficients.
    atoms (float 2D): Matrix of atoms extracted.
    residual (float 1D): Residual after termination main loop.
    indices (float 1D): Column indices of the selected atoms.
    """

    # Limit the maximum sparsity to the number of atoms, i.e., all atoms may
    # be used.
    if K <= 0 or K > dictionary.shape[1]:
        K = dictionary.shape[1] // N * N

    # Store the residual and dictionary as contiguous arrays in memory to speed
    # up the calculation of the dot product.
    signal = np.ascontiguousarray(signal)
    residual = np.ascontiguousarray(signal)
    dictionary = np.ascontiguousarray(dictionary)

    # Initialize coefficients, atoms, and errors
    m = len(signal)
    atoms = np.full((m, K), 0., dtype=dictionary.dtype)
    indices = np.full(K, 0., dtype=np.int32)
    all_inds = np.arange(dictionary.shape[1], dtype=np.int32)

    # The original signal's L2-norm is constant, so we calculate it before of
    # the loop to avoid calculating it for each iteration.
    signal_l2 = np.sqrt(np.sum(np.square(signal)))

    # Begin the main loop.
    k = 0
    while np.sqrt(np.sum(np.square(residual))) / signal_l2 > maxerr:
        # Determine the N most correlated atoms to the current residual and
        # determine their indices.
        dot_product = np.dot(residual, dictionary)
        inds = np.abs(dot_product).argsort()[-N:][::-1]
        indices[k*N:(k+1)*N] = all_inds[inds]

        # Store selected atoms.
        atoms[:, k*N:(k+1)*N] = dictionary[:, inds]

        # Update residual.
        # If dictionary contains complex values, use np.conj(A).T instead of A.T
        A = np.ascontiguousarray(atoms[:, :(k+1)*N])
        estimate = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), signal)
        residual = signal - np.dot(A, estimate)

        # Remove selected atoms from dictionary.
        dictionary = np.ascontiguousarray(delete_column(dictionary, inds))
        all_inds = np.delete(all_inds, inds)

        # If we have reached the desired sparsity or there are no atoms left
        # in the dictionary, then terminate the main loop.
        k += 1
        if k*N >= K or dictionary.size == 0:
            break

    indices = indices[:k*N]
    atoms = np.ascontiguousarray(atoms[:, :k*N])
    coefficients = np.dot(np.dot(np.linalg.inv(np.dot(atoms.T, atoms)),
        atoms.T), signal)

    return coefficients, atoms, residual, indices