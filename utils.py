import numpy as np
import matplotlib.pyplot as plt

def plot_approximation(
    coefficients: np.ndarray,
    atoms: np.ndarray,
    max_order: int = None,
    signal: np.ndarray = None, 
    residual: np.ndarray = None,
    **kwargs
):
    if max_order is None:
        max_order = atoms.shape[1]
    elif max_order > atoms.shape[1]:
        max_order = atoms.shape[1]
    
    if signal is not None:
        if residual is not None:
            fig, ax = plt.subplots(max_order + 2, 1, **kwargs)
        else:
            fig, ax = plt.subplots(max_order + 1, **kwargs)
        ax[0].plot(signal[0], label="original signal")
        ax[0].grid(True)
        ax[0].legend(loc="upper right")
        for i in range(max_order):
            ax[i+1].plot(coefficients[:, i] * atoms[:, i],
                label=f"approx order {i+1}")
            ax[i+1].grid(True)
            ax[i+1].legend(loc="upper right")
        if residual is not None:
            ax[max_order+1].plot(residual[0],
                label=f"residual order {atoms.shape[1]}")
            ax[max_order+1].grid(True)
            ax[max_order+1].legend(loc="upper right")
    else:
        if residual is not None:
            fig, ax = plt.subplots(max_order + 1, 1, **kwargs)
        else:
            fig, ax = plt.subplots(max_order, 1, **kwargs)
        for i in range(max_order):
            ax[i].plot(coefficients[:, i] * atoms[:, i],
                label=f"approx order {i+1}")
            ax[i].grid(True)
            ax[i].legend(loc="upper right")
        if residual is not None:
            ax[max_order].plot(residual[0],
                label=f"residual order {atoms.shape[1]}")
            ax[max_order].grid(True)
            ax[max_order].legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def plot_reconstructed_signal(
    reconstructed_signal: np.ndarray,
     signal: np.ndarray = None,
     split: bool = False,
    **kwargs
):
    if signal is not None and split:
        fig, ax = plt.subplots(2, 1, **kwargs)
        ax[0].plot(signal[0], color="#1f77b4", linestyle='-',
            label="original signal")
        ax[1].plot(reconstructed_signal[0], color="#ff7f0e", linestyle='-',
            label="reconstructed signal")
        for i in range(2):
            ax[i].legend(loc="upper right")
            ax[i].grid(True)
    elif signal is not None and not split:
        fig, ax = plt.subplots(1, 1, **kwargs)
        ax.plot(signal[0], color="#1f77b4", linestyle='-',
            label="original signal")
        ax.plot(reconstructed_signal[0], color="#ff7f0e", linestyle='--',
            label="reconstructed signal")
        ax.grid(True)
        ax.legend(loc="upper right")
    else:
        fig, ax = plt.subplots(1, 1, **kwargs)
        ax.plot(reconstructed_signal[0], color="#ff7f0e", linestyle='-',
            label="reconstructed signal")
        ax.grid(True)
        ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()