
import numpy as np
import functions as fct
import matplotlib.pyplot as plt






steps = 1000
lr = 0.001
tasks = ["GoNogo-v0","PerceptualDecisionMaking-v0"]

net, loss_trajectory = fct.train_simultaneous_integration(tasks,steps,lr)


HW = net.rnn.h2h.weight.detach().numpy()

plt.figure()
plt.imshow(HW)
plt.savefig("figures/HW_integration.png")


