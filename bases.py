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

def erb(fc: float):
    """
    Only valid for frequencies in the range 0 Hz to 10 kHz.
    """
    return 24.7 * (4.37 * fc / 1000 + 1)

def gammatone_function(
    N: int,
    fc: int,
    fs: int = 16000,
    l: int = 4,
    b: float = 1.019,
    dtype: type = np.float32
):
    nT = np.arange(0, N) / fs
    return np.power(nT, l-1) * np.exp(-2*np.pi*b*erb(fc)*nT) * np.cos(
        2*np.pi*fc*nT, dtype=dtype)

def gammatone_matrix(
    N: int,
    fc: int,
    fs: int = 16000,
    step: int = 8,
    l: int = 4,
    b: float = 1.019,
    dtype: type = np.float32
):  
    """
    Gammatone matrix. Please don't use this, it isn't optimized yet. It already
    works so I dind't want to touch it anymore.
    """
    tc = (l - 1) / (2 * np.pi * b * erb(fc))
    centers = np.arange(-int(tc*fs), int(N - tc*fs) + step, step)
    basis = []
    for center in centers:
        if center < 0:
            gammatone = gammatone_function(N+abs(center), fc, fs, l, b, dtype)
            atom = gammatone[abs(center):]
        else:
            gammatone = gammatone_function(N, fc, fs, l, b, dtype)
            atom = np.full(N, 0., dtype)
            atom[center:] = gammatone[:N-center]
        basis.append(atom) 
    basis = np.array(basis, dtype=dtype).T
    return basis