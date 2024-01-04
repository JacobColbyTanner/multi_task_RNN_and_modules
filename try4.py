
import numpy as np
import matplotlib.pyplot as plt


A = np.load("data/staged_development/accuracy.npy")

current_iter = 62

fix_modules = np.mean(A[0,0,0:current_iter,:].squeeze(),axis = 0)
no_fix_modules = np.mean(A[1,0,0:current_iter,:].squeeze(),axis = 0) 
fix_no_modules = np.mean(A[0,1,0:current_iter,:].squeeze(),axis = 0) 
no_fix_no_modules = np.mean(A[1,1,0:current_iter,:].squeeze(),axis = 0)

plt.figure()
plt.plot(no_fix_modules, label='no fix modules')
plt.plot(fix_no_modules, label='fix no modules')
plt.plot(no_fix_no_modules, label='no fix no modules')
plt.plot(fix_modules, label='fix modules')
plt.xlabel("steps")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("figures/development_accuracy.png")
