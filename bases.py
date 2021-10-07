import math
import numpy as np

def dct2_basis(N: int, dtype: type = np.float32):
    """
    Discrete cosine transfor-II (DCT-II) orthogonal basis.
    """
    basis = []
    for k in range(N):
        if k == 0:
            basis.append(np.full(N, math.sqrt(1 / N)))
        else:
            basis.append(math.sqrt(2 / N) * np.cos(math.pi / N * (np.arange(N) \
                + 1 / 2) * k, dtype=dtype))
    return np.array(basis, dtype=dtype).T

def dst1_basis(N: int, dtype: type = np.float32):
    """
    Discrete sine transform-I (DST-I) orthogonal basis.
    """
    basis = []
    for k in range(N):
        basis.append(np.sqrt(2 / (N + 1)) * np.sin(np.pi / (N + 1) * (
            np.arange(N) + 1) * (k + 1), dtype=dtype))
    return np.array(basis, dtype=dtype).T

def dst2_basis(N: int, dtype: type = np.float32):
    """
    Discrete sine transform-II (DST-II) basis.
    """
    basis = []
    for k in range(N):
        basis.append(np.sin(math.pi / N * (np.arange(N) \
            + 1 / 2) * (k + 1), dtype=dtype))
    return np.array(basis, dtype=dtype).T

def sin_basis(N: int, dtype: type = np.float32):
    """
    Sine subdictionary.
    """
    basis = []
    for k in range(1, math.ceil(N / 2) + 1):
        basis.append(np.sin(2 * np.pi * k * np.linspace(0, 1, N), dtype=dtype))
    return np.array(basis).T

def cos_basis(N: int, dtype: type = np.float32):
    """
    Cosine subdictionary.
    """
    basis = []
    for k in range(1, math.ceil(N / 2) + 1):
        basis.append(np.cos(2 * np.pi * k * np.linspace(0, 1, N), dtype=dtype))
    return np.array(basis).T

def poly_basis(N: int, dtype: type = np.float32):
    """
    Polynomial subdictionary.
    """
    basis = []
    for n in range(1, 21):
        basis.append(np.power(np.linspace(0, 1, N, dtype=dtype), n - 1))
    return np.array(basis).T

def kd_basis(N: int, dtype: type = np.float32):
    """
    Shifted Kronecker delta subdictionary.
    """
    return np.eye(N, dtype=dtype)

def dft_basis(N: int):
    """
    Discrete Fourier transform (DFT) orthogonal basis.
    """
    basis = []
    for k in range(N):
        basis.append(np.sqrt(1 / N) * np.exp(-2j*np.pi*k*np.arange(N)/N))
    return np.array(basis).T

def dht_basis(N: int, dtype: type = np.float32):
    """
    Discrete Hartley transform (DHT) orthogonal basis.
    """
    basis = []
    for k in range(N):
        basis.append(np.sqrt(2 / N) * np.cos(2 * np.pi * np.arange(N) * k / N \
            - np.pi / 4, dtype=dtype))
    return np.array(basis).T