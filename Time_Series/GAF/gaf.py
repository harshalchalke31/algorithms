import numpy as np
import matplotlib.pyplot as plt

def min_max_normalize(signal:np.array):
    min_value = np.min(signal)
    max_value = np.max(signal)

    return (2 * (signal-min_value)/(max_value-min_value)) - 1

def GAF(signal:np.array,summation:bool=True):

    normalized_signal = min_max_normalize(signal=signal)
    polar_signal = np.arccos(normalized_signal)

    if summation == False:
        return np.sin(polar_signal[:,None] - polar_signal[None,:])
    else:
        return np.cos(polar_signal[:,None] + polar_signal[None,:])

def plot_gaf(gaf, title:str='Gramian Angular Field'):
    plt.figure(figsize=(4, 4))
    plt.imshow(gaf, cmap='rainbow', origin='upper')
    plt.title(title)
    plt.colorbar()
    plt.show()