# Define networks
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import math
    

class CTRNN(nn.Module):
    """Continuous-time RNN.

    Parameters:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
        dt: discretization time step in ms. 
            If None, dt equals time constant tau

    Inputs:
        input: tensor of shape (seq_len, batch, input_size)
        hidden: tensor of shape (batch, hidden_size), initial hidden activity
            if None, hidden is initialized through self.init_hidden()
        
    Outputs:
        output: tensor of shape (seq_len, batch, hidden_size)
        hidden: tensor of shape (batch, hidden_size), final hidden activity
    """

    def __init__(self, input_size, hidden_size, dt=None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha

        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size)

    def recurrence(self, input, hidden):
        """Run network for one time step.
        
        Inputs:
            input: tensor of shape (batch, input_size)
            hidden: tensor of shape (batch, hidden_size)
        
        Outputs:
            h_new: tensor of shape (batch, hidden_size),
                network activity at the next time step
        """
        h_new = torch.relu(self.input2h(input) + self.h2h(hidden))
        h_new = hidden * (1 - self.alpha) + h_new * self.alpha
        return h_new

    def forward(self, input, hidden=None):
        """Propogate input through the network."""
        
        # If hidden activity is not provided, initialize it
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)

        # Loop through time
        output = []
        input_projection = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden)
            input_projection.append(self.input2h(input[i]))

        # Stack together output from all time steps
        output = torch.stack(output, dim=0)  # (seq_len, batch, hidden_size)
        input_projection = torch.stack(input_projection, dim=0)  
        
        return output, hidden, input_projection


class RNNNet(nn.Module):
    """Recurrent network model.

    Parameters:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size
    
    Inputs:
        x: tensor of shape (Seq Len, Batch, Input size)

    Outputs:
        out: tensor of shape (Seq Len, Batch, Output size)
        rnn_output: tensor of shape (Seq Len, Batch, Hidden size)
    """
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()

        # Continuous time RNN
        self.rnn = CTRNN(input_size, hidden_size, **kwargs)
        
        # Add an output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_output, hidden, input_projection = self.rnn(x)
        out = self.fc(rnn_output)
        return out, rnn_output
    
    
    










# Define networks
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import math
import neurogym as ngym
import numpy as np
import torch.nn.utils.prune as prune
import gym  # package for RL environments
import torch.optim as optim
import time
import matplotlib.pyplot as plt






def train_multitask2(tasks,steps,mask,lr,randomize_task_order):
    #total possible tasks is 14 with this function
    """function to train the model on multiple tasks.
   
    Args:
        net: a pytorch nn.Module module
        dataset: a dataset object that when called produce a (input, target output) pair
   
    Returns:
        net: network object after training
    """
    if torch.cuda.is_available(): 
        dev = "cuda:0" 
        print(dev)
    else: 
        dev = "cpu" 
        print(dev)
    device = torch.device(dev) 
   

    

    # set tasks
    dt = 100

    num_tasks = len(tasks)
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
            o1[task] = env.action_space.n
        except:
            o1[task] = env.action_space.shape[0]


    input_size = int(np.sum(i1))
    output_size = int(np.sum(o1))
    
    
    hidden_size = mask.shape[0]
   
   
    net = RNNNet(input_size=input_size, hidden_size=hidden_size,
             output_size=output_size, dt=dt)
             
    net.to(device)
   
        #apply pruning mask
    mask = torch.from_numpy(mask).to(device)
    apply = prune.custom_from_mask(net.rnn.h2h, name = "weight", mask = mask)
   
    # Use Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    running_loss = 0
    running_acc = 0
    start_time = time.time()
    # Loop over training batches
    #print('Training network...')
    
   
    loss_trajectory = np.zeros([steps])
    
    
   
    for i in range(steps):
        # Generate input and target(labels) for all tasks, then concatenate and convert to pytorch tensor
        
        
        inputs1 = {}
        labels1 = {}
        for task in range(num_tasks):
            data = dataset1[task]
            inputs1[task], labels1[task] = data()
            
       
       
       
       
       

        #keep track of number of output neurons while stacking and change output labels so they are unique
        ####need to change labels so that they correspond to proper neuron output
        num_out_cumsum = 0
        for task in range(num_tasks):
       
            if task != 0:
       
       
                num_out_cumsum = num_out_cumsum + o1[task-1]

       
                idd = labels1[task] > 0
           
           
               
                labels1[task][idd] = labels1[task][idd]+num_out_cumsum
               
        rand_list = np.random.permutation(num_tasks)
        #now stack them
                #Here is where you could change the order of the tasks
        #Currently random ordering
       
        if randomize_task_order == 1:
            for task in range(num_tasks):

                if task == 0:
                    labels = labels1[rand_list[task]]
                else:
                    labels = np.concatenate((labels, labels1[rand_list[task]]), axis=0)


        else:
            for task in range(num_tasks):

                if task == 0:
                    labels = labels1[task]
                else:
                    labels = np.concatenate((labels, labels1[task]), axis=0)


       
        labels = torch.from_numpy(labels.flatten()).type(torch.long).to(device)
        #plt.hist(labels)
        #plt.show()
   
        ###Need to concatenate inputs along sequence length so that the tasks are given to the network sequentially
        #make same size on axis 3
        before = 0
        after = int(np.asarray(input_size)-i1[0])

        inputs1[0] = np.pad(inputs1[0],((0,0),(0,0),(before,after)))

       
        for task in range(num_tasks):
           
            if task != 0:
                before += int(i1[task-1])
                after = int(after-i1[task])
               
                inputs1[task] = np.pad(inputs1[task],((0,0),(0,0),(before,after)))

               

        #Here is where you could change the order of the tasks
        #Currently random ordering
       
        if randomize_task_order == 1:
           

            for task in range(num_tasks):
                if task == 0:
                    inputs = inputs1[rand_list[task]] #rand_list[task]
                else:
                    inputs = np.concatenate((inputs,inputs1[rand_list[task]]), axis=0)

        else:
            for task in range(num_tasks):
                if task == 0:
                    inputs = inputs1[task] #rand_list[task]
                else:
                    inputs = np.concatenate((inputs,inputs1[task]), axis=0)

           
        inputs = torch.from_numpy(inputs).type(torch.float).to(device)
       
       

       

       
        # boiler plate pytorch training:
        optimizer.zero_grad()   # zero the gradient buffers
        output, _ = net(inputs)
        # Reshape to (SeqLen x Batch, OutputSize)
        output = output.view(-1, output_size).to(device)
       
        #print("output shape: " + str(output.shape))
        #print("labels shape: " + str(labels.shape))
       
        
        loss = criterion(output, labels)
        
        loss.backward()
        optimizer.step()    # Does the update

        loss_trajectory[i] = loss.item()
       
        # Compute the running loss every 100 steps
        running_loss += loss.item()
        if i % 100 == 99:
            running_loss /= 100
            print('Step {}, Loss {:0.4f}, Time {:0.1f}s'.format(
                i+1, running_loss, time.time() - start_time))
            running_loss = 0
    return net, loss_trajectory







def train_simultaneous_integration(tasks,steps,mask,lr):
    #total possible tasks is 14 with this function
    """function to train the model on multiple tasks.
   
    Args:
        net: a pytorch nn.Module module
        dataset: a dataset object that when called produce a (input, target output) pair
   
    Returns:
        net: network object after training
    """
   
    if torch.cuda.is_available(): 
        dev = "cuda:0" 
        print(dev)
    else: 
        dev = "cpu" 
        print(dev)
    device = torch.device(dev) 
    # set tasks
    dt = 100
    #tasks = ["SineWavePred-v0","GoNogo-v0","PerceptualDecisionMaking-v0"]
    
    #tasks = ["GoNogo-v0","PerceptualDecisionMaking-v0"]
    num_tasks = len(tasks)
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
    #to create a tensor object that is AxBxCxD.. etc
    output_size = int(np.prod(o1))
    hidden_size = mask.shape[0]
   
   
    net = RNNNet(input_size=input_size, hidden_size=hidden_size,
             output_size=output_size, dt=dt)
   
    net.to(device)
        #apply pruning mask
    mask = torch.from_numpy(mask).to(device)
    apply = prune.custom_from_mask(net.rnn.h2h, name = "weight", mask = mask)
   
    # Use Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    
    start_time = time.time()



    num_param = 0
    num_param += net.rnn.input2h.weight.cpu().detach().numpy().size
    num_param += net.rnn.input2h.bias.cpu().detach().numpy().size
    num_param += net.rnn.h2h.weight.cpu().detach().numpy().size
    num_param += net.rnn.h2h.bias.cpu().detach().numpy().size
    num_param += net.fc.weight.cpu().detach().numpy().size
    num_param += net.fc.bias.cpu().detach().numpy().size

   
    loss_trajectory = np.zeros([steps])
    weight_trajectory = np.zeros((steps+1,num_param))
    running_loss = 0
    for i in range(steps):
        # Generate input and target(labels) for all tasks, then concatenate and convert to pytorch tensor
        weight_trajectory[i,:] = weight_shape_to_flat(net)
        
        
        timing = {
            'fixation': 100,
            'stimulus': 2000,
            'delay': np.random.randint(200,high=600),
            'decision': 100}
        kwargs = {'dt': dt, 'timing':timing}
        
        for task in range(num_tasks):
            dataset1[task] = ngym.Dataset(tasks[task], env_kwargs=kwargs, batch_size=16,
                       seq_len=seq_len)
            data = dataset1[task]
            inputs1, labels1 = data()
            
            if task == 0:
                inputs = inputs1
            else:
                inputs = np.concatenate((inputs,inputs1), axis=2)
                
            if task == 0:
                labels = labels1
                labels = np.expand_dims(labels, axis=2)
            else:
                labels1 = np.expand_dims(labels1, axis=2)
                labels = np.concatenate((labels,labels1), axis=2)
            
                
        inputs = torch.from_numpy(inputs).type(torch.float).to(device)
        
        
        new_labels = np.zeros(labels1.shape)
        for ii in range(labels.shape[0]):
            for batch in range(labels.shape[1]):
                L = labels[ii,batch,:]
                
                

                label_tensor = np.zeros(tuple(o1.astype(int)))
                label_tensor[tuple(L)] = 1
                
                label_tensor = label_tensor.flatten()
                ind = np.nonzero(label_tensor)
                new_labels[ii,batch] = ind[0]
            
            
        labels = torch.from_numpy(new_labels.flatten()).type(torch.long).to(device)

        #for task in range(num_tasks):
            

        #plt.plot(labels)
        #plt.show()
        
       
       
        # boiler plate pytorch training:
        optimizer.zero_grad()   # zero the gradient buffers
        output, _ = net(inputs)
        # Reshape to (SeqLen x Batch, OutputSize)
        output = output.view(-1, output_size)
       
        #print("output shape: " + str(output.shape))
        #print("labels shape: " + str(labels.shape))
       
        
        loss = criterion(output, labels)
        
        loss.backward()
        optimizer.step()    # Does the update

        loss_trajectory[i] = loss.item()
       
        # Compute the running loss every 100 steps
        running_loss += loss.item()
        if i % 100 == 99:
            running_loss /= 100
            print('Step {}, Loss {:0.4f}, Time {:0.1f}s'.format(
                i+1, running_loss, time.time() - start_time))
            running_loss = 0
            
    return net, loss_trajectory, weight_trajectory

