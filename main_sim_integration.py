
import numpy as np
import functions as fct
import matplotlib.pyplot as plt
import time





steps = 100
lr = 0.001
tasks = ["GoNogo-v0","PerceptualDecisionMaking-v0"]

net, loss_trajectory = fct.train_simultaneous_integration(tasks,steps,lr)


HW = net.rnn.h2h.weight.detach().numpy()

plt.figure()
plt.imshow(HW)
plt.colorbar()
plt.savefig("figures/HW_integration.png")


np.save("data/HW.npy", HW)

counts, bins = np.histogram(HW,bins = 1000)

plt.figure()
plt.stairs(counts, bins)
plt.savefig("figures/HW_histogram.png")



hidden_size = 100
batch_size = 20
num_runs = 1

#DSS deletion step size
dss = 0.0001
steps_delete = 100

step = dss

A = np.zeros(steps_delete)
HWs = []

for S in range(steps_delete):
    print("deleting edges: ",S)
    start_time = time.time()
    mask = np.absolute(HW) < step
    step = step+dss
    lesioned_model = fct.lesion_rnn_mask(net,mask)
    

    accuracy,accuracy_task = fct.get_accuracy_integration(net,tasks,hidden_size)

    A[S] = np.mean(accuracy)
    HWs.append(lesioned_model.rnn.h2h.weight.detach().numpy())

    end_time = time.time() - start_time

    print("time: ", end_time)

plt.figure()
plt.plot(A)
plt.xlabel("deletion weights below:")
plt.ylabel("accuracy")
plt.savefig("figures/accuracy_deletion.png")









