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
    return np.array(basis, dtype=dtype)

def dst1_basis(N: int, dtype: type = np.float32):
    """
    Discrete sine transform-I (DST-I) orthogonal basis.
    """
    basis = []
    for k in range(N):
        basis.append(np.sqrt(2 / (N + 1)) * np.sin(np.pi / (N + 1) * (
            np.arange(N) + 1) * (k + 1), dtype=dtype))
    return np.array(basis, dtype=dtype)

def dst2_basis(N: int, dtype: type = np.float32):
    """
    Discrete sine transform-II (DST-II) basis.
    """
    basis = []
    for k in range(N):
        basis.append(np.sin(math.pi / N * (np.arange(N) \
            + 1 / 2) * (k + 1), dtype=dtype))
    return np.array(basis, dtype=dtype)

def sin_basis(N: int, dtype: type = np.float32):
    """
    Sine subdictionary.
    """
    basis = []
    for k in range(1, math.ceil(N / 2) + 1):
        basis.append(np.sin(2 * np.pi * k * np.linspace(0, 1, N), dtype=dtype))
    return np.array(basis)

def cos_basis(N: int, dtype: type = np.float32):
    """
    Cosine subdictionary.
    """
    basis = []
    for k in range(1, math.ceil(N / 2) + 1):
        basis.append(np.cos(2 * np.pi * k * np.linspace(0, 1, N), dtype=dtype))
    return np.array(basis)

def poly_basis(N: int, dtype: type = np.float32):
    """
    Polynomial subdictionary.
    """
    basis = []
    for n in range(1, 21):
        basis.append(np.power(np.linspace(0, 1, N, dtype=dtype), n - 1))
    return np.array(basis)

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
    return np.array(basis)

def dht_basis(N: int, dtype: type = np.float32):
    """
    Discrete Hartley transform (DHT) orthogonal basis.
    """
    basis = []
    for k in range(N):
        basis.append(np.sqrt(2 / N) * np.cos(2 * np.pi * np.arange(N) * k / N \
            - np.pi / 4, dtype=dtype))
    return np.array(basis)

def gabor_atom(
    N: int,
    w: float,
    s: float,
    u: int,
    theta: float = 0
):
    return np.sqrt(1 / s) * np.exp(-np.pi*np.square(np.arange(N)-u)/(s*s)) \
        * np.cos(2*np.pi*w*(np.arange(N)-u) - theta)

def gabor_basis(
    N: int,
    i_vals: list = range(1, 36),
    p_vals: list = range(1, 9),
    u_step: int = 64,
    dtype: type = np.float32
):
    """
    Gabor basis.
    """
    K = 0.5 * i_vals[-1]**(-2.6)

    a_freqs = [K*i**2.6 for i in i_vals]
    scales = [2**p for p in p_vals]
    time_shifts = range(0, N, u_step)

    dictionary = []

    for i in range(len(a_freqs)):
        for j in range(len(scales)):
            for k in range(len(time_shifts)):
                dictionary.append(gabor_atom(N, a_freqs[i], scales[j],
                    time_shifts[k]))

    return np.array(dictionary, dtype)