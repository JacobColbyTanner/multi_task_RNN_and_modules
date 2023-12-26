


import numpy as np
import matplotlib.pyplot as plt





OSQ = np.load("data/OSQ.npy")

OSQ = np.mean(OSQ[:,:,:],axis=0).squeeze()

plt.figure()
plt.imshow(OSQ)
plt.colorbar()
plt.savefig("figures/OSQ.png")

phase_dist = np.load("data/phase_distance.npy")
phase_dist = np.mean(phase_dist[:,:,:],axis=0).squeeze()

plt.figure()
plt.imshow(phase_dist)
plt.colorbar()
plt.savefig("figures/phase_dist.png")


HW = np.load("data/HW.npy")

counts, bins = np.histogram(HW,bins = 1000)

plt.figure()
plt.stairs(counts, bins)
plt.savefig("figures/HW_histogram.png")