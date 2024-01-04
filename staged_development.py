import torch
import numpy as np
import functions as fct



tasks = ["GoNogo-v0","PerceptualDecisionMaking-v0"]
steps = 1000
lr = 0.001
num_iter = 100


SQ = np.zeros((2,2,num_iter))
OSQ = np.zeros((2,2,num_iter))
loss = np.zeros((2,2,num_iter,steps))
accuracy = np.zeros((2,2,num_iter,steps))

for iter in range(num_iter):
    print("----------------------------------------iteration: ", iter)
    np.save("data/staged_development/SQ.npy",SQ)
    np.save("data/staged_development/OSQ.npy",OSQ)
    np.save("data/staged_development/loss.npy",loss)
    np.save("data/staged_development/accuracy.npy",accuracy)
    
    for F in range(2):
        for SC in range(2):
            if F == 0:
                freeze = True
            else:
                freeze = False

            if SC == 0:
                off_block = 0.125
                thalamic_bias = 2
            else:
                off_block = 0
                thalamic_bias = 0

            
            net, SQ[F,SC,iter], OSQ[F,SC,iter], mask = fct.SC_modules_thalamic_bias2(off_block, thalamic_bias,tasks)
            net2, loss[F,SC,iter,:], accuracy[F,SC,iter,:] = fct.train_simultaneous_integration_staged(net,tasks,steps,lr,mask,freeze)

