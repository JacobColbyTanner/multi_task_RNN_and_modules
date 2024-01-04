
import numpy as np
import matplotlib.pyplot as plt


A = np.load("data/A_diff_mean.npy")

A = np.mean(A[0:70,:,:],axis=0).squeeze()

plt.figure()
plt.imshow(A)
plt.colorbar()
plt.savefig("figures/A_diff_mean.png")

