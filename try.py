


import numpy as np
import matplotlib.pyplot as plt


def remove_outliers(data, m=2):
    """
    Replace outliers in a numpy array with NaN.
    
    Parameters:
    data (np.array): The input numpy array. Can be multi-dimensional.
    m (float): The number of standard deviations from the mean. 
               Points lying beyond this threshold will be considered outliers.
               
    Returns:
    np.array: The array with outliers replaced by NaN.
    """
    mean = np.mean(data)
    std_dev = np.std(data)
    return np.where(abs(data - mean) < m * std_dev, data, np.nan)



OSQ = np.load("data/OSQ.npy")

OSQ = np.mean(OSQ[:,:,:],axis=0).squeeze()

plt.figure()
plt.imshow(OSQ)
plt.colorbar()
plt.savefig("figures/OSQ.png")


SQ = np.load("data/SQ.npy")

SQ = np.mean(SQ[:,:,:],axis=0).squeeze()

plt.figure()
plt.imshow(SQ)
plt.colorbar()
plt.savefig("figures/SQ.png")

phase_dist = np.load("data/phase_distance.npy")
PD = remove_outliers(phase_dist[:,:,:])
phase_dist = np.nanmean(PD,axis=0).squeeze()

plt.figure()
plt.imshow(phase_dist)
plt.colorbar()
plt.savefig("figures/phase_dist.png")


HW = np.load("data/HW.npy")

counts, bins = np.histogram(HW,bins = 1000)

plt.figure()
plt.stairs(counts, bins)
plt.savefig("figures/HW_histogram.png")