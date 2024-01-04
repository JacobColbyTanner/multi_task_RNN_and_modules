

import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import functions as fct
from scipy.io import savemat
import neurogym as ngym

tasks = ["GoNogo-v0","PerceptualDecisionMaking-v0"]
num_tasks = len(tasks)
dt = 100
kwargs = {'dt': dt}
seq_len = 100

# Make supervised datasets
i1 = np.zeros([num_tasks])
o1 = np.zeros([num_tasks])
dataset1 = {}
for task in range(num_tasks):
    
    dataset1[task] = ngym.Dataset(tasks[task], env_kwargs=kwargs, batch_size=16,
                    seq_len=seq_len)
    
                #get input and output sizes for different tasks
    env = dataset1[task].env
    i1[task] = env.observation_space.shape[0]
    try:
        o1[task] = int(env.action_space.n)
    except:
        o1[task] = int(env.action_space.shape[0])
input_size = int(np.sum(i1))
output_size = int(np.prod(o1))
hidden_size = 100


net = fct.RNNNet(input_size=input_size, hidden_size=hidden_size,
        output_size=output_size, dt=dt)


net.load_state_dict(torch.load('data/net_integration_weights.pth'))


activity, output, inputs, labels = fct.get_activity_integration(net,tasks)

mdic = {"activity":activity.detach().numpy(), "outputs":output.detach().numpy(), "inputs": inputs.detach().numpy(),"labels":labels}
savemat("/N/project/networkRNNs/integration_task_activity.mat",mdic)

HWs = np.load("data/HWs.npy")

mdic2 = {"HWs":HWs}
savemat("/N/project/networkRNNs/integration_task_HW.mat",mdic2)
