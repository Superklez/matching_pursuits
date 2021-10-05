import math
import numpy as np

def dct_basis(N: int, dtype: type = np.float32):
    '''
    Discrete cosine transfor-II (DCT-II) orthogonal basis.
    '''
    basis = []
    for k in range(N):
        if k == 0:
            basis.append(np.full(N, math.sqrt(1 / N)))
        else:
            basis.append(math.sqrt(2 / N) * np.cos(math.pi / N * (np.arange(N) \
                + 1 / 2) * k, dtype=dtype))
    return np.array(basis, dtype=dtype).T

def sin_basis(N: int, fs: int = 16000):
    '''
    Sine subdictionary.
    '''
    basis = []
    for k in range(1, math.ceil(N / 2) + 1):
        atom = np.sin(2 * np.pi * k * np.linspace(0, 1, fs, endpoint=False))
        while len(atom) < N:
            atom = np.tile(atom, 2)
        atom = atom[:N]
        basis.append(atom)
    return np.array(basis).T

def cos_basis(N: int, fs: int = 16000):
    '''
    Cosine subdictionary.
    '''
    basis = []
    for k in range(1, math.ceil(N / 2) + 1):
        atom = np.cos(2 * np.pi * k * np.linspace(0, 1, fs, endpoint=False))
        while len(atom) < N:
            atom = np.tile(atom, 2)
        atom = atom[:N]
        basis.append(atom)
    return np.array(basis).T

def poly_basis(t: int, fs: int = 16000):
    '''
    Kronecker delta subdictionary.
    '''
    basis = []
    for n in range(1, 21):
        atom = np.power(np.linspace(0, 1, fs, endpoint=False), n - 1)
        while len(atom) < t:
            atom = np.tile(atom, 2)
        atom = atom[:t]
        basis.append(atom)
    return np.array(basis).T

def kd_basis(N: int, dtype: type = np.float32):
    '''
    Shifted Kronecker delta subdictionary.
    '''
    return np.eye(N, dtype=dtype)