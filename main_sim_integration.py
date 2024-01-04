
import numpy as np
import functions as fct
import matplotlib.pyplot as plt
import time
import torch





accuracy_goal = 0.95
lr = 0.0001
tasks = ["GoNogo-v0","PerceptualDecisionMaking-v0"]

net, loss_trajectory = fct.train_simultaneous_integration(tasks,accuracy_goal,lr)


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
dss = 0.001
steps_delete = 200

step = dss

A = np.zeros(steps_delete+1)

HWs = []

accuracy = fct.get_accuracy_integration(net,tasks,hidden_size)
print("real accuracy: ",np.mean(accuracy))
A[0] = np.mean(accuracy)
all_steps = np.zeros(steps_delete+1)
all_steps[0] = step
num_del = np.zeros((steps_delete+1)/10)
num_del[0] = 0
iii = 0
for S in range(steps_delete):
    print("deleting edges: ",S,flush=True)
    start_time = time.time()
    mask = np.absolute(HW) < step
    if S%10 == 0:
        iii= iii+1
        num_del[iii] = np.sum(mask)
        print("num deleted: ", num_del[S+1])
    step = step+dss
    lesioned_model = fct.lesion_rnn_mask(net,mask)
    
    all_steps[S+1] = step

    accuracy = fct.get_accuracy_integration(lesioned_model,tasks,hidden_size)
    print("next_accuracy: ",np.mean(accuracy))
    A[S+1] = np.mean(accuracy)
    HWs.append(lesioned_model.rnn.h2h.weight.detach().numpy())

    end_time = time.time() - start_time

    #print("time: ", end_time)

plt.figure()
plt.plot(A)
plt.xlabel("# edges deleted")
plt.ylabel("accuracy")
plt.xticks(np.arange(0, S+2),num_del)
plt.savefig("figures/accuracy_deletion.png")

#save weights
np.save("data/HWs.npy",HWs)
#save network
torch.save(net.state_dict(), 'data/net_integration_weights.pth')








