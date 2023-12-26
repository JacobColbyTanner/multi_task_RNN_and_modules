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






def train_multitask2(tasks,steps,mask,lr,randomize_task_order,input_weights):
    
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
   
    tensor = torch.tensor(input_weights, dtype=torch.float32)

    with torch.no_grad():
        net.rnn.input2h.weight = torch.nn.Parameter(tensor)

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


import torch

import numpy as np

def place_nodes_on_sphere(N, R):
    # Generate random angles theta and phi
    theta = np.random.uniform(0, 2*np.pi, N)
    phi = np.arccos(np.random.uniform(-1, 1, N))

    # Convert spherical coordinates to Cartesian coordinates
    x = R * np.sin(phi) * np.cos(theta)
    y = R * np.sin(phi) * np.sin(theta)
    z = R * np.cos(phi)

    # Initialize the distance matrix
    distance_matrix = np.zeros((N, N))

    # Compute the distance matrix
    for i in range(N):
        for j in range(i + 1, N):
            distance = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2 + (z[i] - z[j])**2)
            distance_matrix[i, j] = distance_matrix[j, i] = distance

    return distance_matrix




def spatial_embed(weight_matrix, se1=0.3, comms_factor=1):
    ''' THIS COMES FROM: Spatially embedded recurrent neural networks 
    reveal widespread links between structural and functional neuroscience findings
    Jascha Achterberg
    Version of SE1 regulariser which combines the spatial and communicability parts in loss function.
    Additional comms_factor scales the communicability matrix.
    The communicability term used here is unbiased weighted communicability:
    Crofts, J. J., & Higham, D. J. (2009). A weighted communicability measure applied to complex brain networks. Journal of the Royal Society Interface, 6(33), 411-414.
    '''
    # Example usage
    N = 100  # Number of nodes
    R =0.5  # Radius of the sphere

    distance_tensor = place_nodes_on_sphere(N, R)
    distance_tensor = torch.from_numpy(distance_tensor)
    # Take absolute of weights
    abs_weight_matrix = torch.abs(weight_matrix)

    # Calculate weighted communicability (as per the reference)
    stepI = torch.sum(abs_weight_matrix, dim=1)
    stepII = torch.pow(stepI, -0.5)
    stepIII = torch.diag_embed(stepII)
    stepIV = torch.matrix_exp(stepIII @ abs_weight_matrix @ stepIII)
    comms_matrix = torch.diag_embed(torch.zeros(stepIV.shape[0]), offset=0, dim1=-2, dim2=-1) + \
                   torch.triu(stepIV, diagonal=1) + torch.tril(stepIV, diagonal=-1)

    # Multiply absolute weights with communicability weights
    comms_matrix = comms_matrix ** comms_factor
    comms_weight_matrix = abs_weight_matrix * comms_matrix

    # Multiply comms weights matrix with distance tensor, scale the mean, and return as loss
    se1_loss = se1 * torch.sum(comms_weight_matrix * distance_tensor)

    return se1_loss

import copy

def train_simultaneous_integration(tasks,steps,lr):
    
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
    hidden_size = 100
   
   
    net = RNNNet(input_size=input_size, hidden_size=hidden_size,
             output_size=output_size, dt=dt)
   
    net.to(device)
    '''
        #apply pruning mask
    mask = torch.from_numpy(mask).to(device)
    apply = prune.custom_from_mask(net.rnn.h2h, name = "weight", mask = mask)
    '''

    # Use Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    
    start_time = time.time()


   
    loss_trajectory = np.zeros([steps])
    
    running_loss = 0
    for i in range(steps):
        # Generate input and target(labels) for all tasks, then concatenate and convert to pytorch tensor
        
        
        
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
        hidden_weights = copy.deepcopy(net.rnn.h2h.weight)
        SE_loss = spatial_embed(hidden_weights, se1=0.3, comms_factor=1)
        loss1 = criterion(output, labels)
        loss = loss1 + SE_loss
        
        loss.backward()
        optimizer.step()    # Does the update

        loss_trajectory[i] = loss.item()
       
        # Compute the running loss every 100 steps
        running_loss += loss.item()
        if i % 100 == 99:
            running_loss /= 100
            print('Step {}, Loss {:0.4f}, Time {:0.1f}s'.format(
                i+1, running_loss, time.time() - start_time))
            print("loss, no distance: ",loss1,flush=True)
            running_loss = 0
            
    return net, loss_trajectory

def get_accuracy_integration(net,tasks,hidden_size):
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
    hidden_size = 100
   
   
    net = RNNNet(input_size=input_size, hidden_size=hidden_size,
             output_size=output_size, dt=dt)
   
    net.to(device)
    '''
        #apply pruning mask
    mask = torch.from_numpy(mask).to(device)
    apply = prune.custom_from_mask(net.rnn.h2h, name = "weight", mask = mask)
    '''


    
 
    accuracy_bunch = []

    steps = 200

    for i in range(steps):
        # Generate input and target(labels) for all tasks, then concatenate and convert to pytorch tensor
        
        
        
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
            


        output, _ = net(inputs)
        # Reshape to (SeqLen x Batch, OutputSize)
        output = output.view(-1, output_size)
       
        # Assuming outputs and labels are already defined:
        predicted_labels = torch.argmax(output, dim=1)

        # Create a mask where predicted_labels are not 0 or 12 (which are fixation outputs that shouldn't be counted)
        fixations = [0]
        masks = []
        for task in range(num_tasks):
            if task > 0:
                fixations.append(int(i1[task-1])+1)
            masks.append(predicted_labels != fixations[task])

        if num_tasks == 3:
            mask = masks[0] & masks[1] & masks[2]
        elif num_tasks == 2:
            mask = masks[0] & masks[1]

        # Use the mask to filter predicted_labels and labels
        filtered_preds = predicted_labels[mask]
        filtered_labels = labels[mask]
        
        
        accuracy_bunch.append(np.nanmean((filtered_preds == filtered_labels).float()))

    print("accuracy bunch: ",accuracy_bunch)
    accuracy = np.nanmean(np.array(accuracy_bunch))
    
   
    return accuracy





            



def get_accuracy(net,tasks,hidden_size,batch_size,num_runs):
    #total possible tasks is 5 with this function

   
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
        dataset1[task] = ngym.Dataset(tasks[task], env_kwargs=kwargs, batch_size=batch_size,
                       seq_len=seq_len)
       
                   #get input and output sizes for different tasks
        env = dataset1[task].env
        i1[task] = env.observation_space.shape[0]
        o1[task] = env.action_space.n


    input_size = int(np.sum(i1))
    output_size = int(np.sum(o1))
    hidden_size = hidden_size
   
    criterion = nn.CrossEntropyLoss()
   
    start_time = time.time()
   
    accuracy = np.zeros([num_runs])
    all_loss = np.zeros([num_runs])
   
    count = 0
    activity_dict = {}
   
    for i in range(num_runs):
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
               
       
        #now stack them        
        for task in range(num_tasks):
           
            if task == 0:
                labels = labels1[task]
            else:
                labels = np.concatenate((labels, labels1[task]), axis=0)
       
       
       
        labels = torch.from_numpy(labels.flatten()).type(torch.long)
   
   
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
        randomize_task_order = 0
        if randomize_task_order == 1:
            rand_list = np.random.permutation(num_tasks)

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

           
        inputs = torch.from_numpy(inputs).type(torch.float)
       
           
 
       
       
        output, rnn_activity = net(inputs)
       
        rnn_activity = rnn_activity[:, 0, :].detach().numpy()
       
        activity_dict[count] = rnn_activity

        if count == 0:
            activity = activity_dict[count]
        else:  
            activity = np.concatenate((activity, activity_dict[count]), axis=0)
           
        count = count+1
       
        # Reshape to (SeqLen x Batch, OutputSize)
        output = output.view(-1, output_size)
       
       
       
        loss = criterion(output, labels)
        all_loss[i] = loss.item()
       
       
        output = output.detach().numpy()
        see = np.argmax(output,axis=1)
       
       
        idx = see == 0
       

       

       
        see = np.delete(see,idx)

       
        labels = np.delete(labels,idx)

        idxx = see == labels

       
        it = np.isclose(see, labels)

        accuracy[i] = np.sum(it)/len(see)
       
       
       
    all_loss = np.nanmean(all_loss)
    mean_accuracy = np.nanmean(accuracy)
   
    return accuracy, mean_accuracy, all_loss, activity



def get_accuracy_per_task(net,tasks,hidden_size,batch_size,num_runs):
    #total possible tasks is 5 with this function

   
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
        dataset1[task] = ngym.Dataset(tasks[task], env_kwargs=kwargs, batch_size=batch_size,
                       seq_len=seq_len)
       
                   #get input and output sizes for different tasks
        env = dataset1[task].env
        i1[task] = env.observation_space.shape[0]
        o1[task] = env.action_space.n


    input_size = int(np.sum(i1))
    output_size = int(np.sum(o1))
    hidden_size = hidden_size
   
    criterion = nn.CrossEntropyLoss()
   
    start_time = time.time()
   
    accuracy_task = np.zeros([3,num_runs])
    
   
    count = 0
    activity_dict = {}
   
    for i in range(num_runs):
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
               
        
        
       
        
        #now stack them 
        '''
        for task in range(num_tasks):
           
            if task == 0:
                labels = labels1[task]
            else:
                labels = np.concatenate((labels, labels1[task]), axis=0)
       
       
       
        labels = torch.from_numpy(labels.flatten()).type(torch.long)
        '''
   
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

               
        '''
        #Here is where you could change the order of the tasks
        #Currently random ordering
        randomize_task_order = 0
        if randomize_task_order == 1:
            rand_list = np.random.permutation(num_tasks)

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

           
        inputs = torch.from_numpy(inputs).type(torch.float)
        '''
           
 
       
        for task in range(3):
            
            inputs = torch.from_numpy(inputs1[task]).type(torch.float)
            output, rnn_activity = net(inputs)

            rnn_activity = rnn_activity[:, 0, :].detach().numpy()

            activity_dict[count] = rnn_activity

            if count == 0:
                activity = activity_dict[count]
            else:  
                activity = np.concatenate((activity, activity_dict[count]), axis=0)

            count = count+1

            # Reshape to (SeqLen x Batch, OutputSize)
            output = output.view(-1, output_size)


            labels = torch.from_numpy(labels1[task].flatten()).type(torch.long)
            
            #loss = criterion(output, labels)



            '''
            output = output.detach().numpy()
            
            see = np.argmax(output,axis=1)
            idx = see == 0
            see = np.delete(see,idx)         
            labels = np.delete(labels,idx)
            idxx = see == labels
            it = np.isclose(see, labels)
            accuracy_task[task,i] = np.sum(it) #/len(see)
            '''
            
         
            # Assuming outputs and labels are already defined:
            predicted_labels = torch.argmax(output, dim=1)

            # Create a mask where predicted_labels are not 0 or 12 (which are fixation outputs that shouldn't be counted)
            fixations = [0]
            masks = []
            for task in range(num_tasks):
                if task > 0:
                    fixations.append(int(i1[task-1])+1)
                masks.append(predicted_labels != fixations[task])

            if num_tasks == 3:
                mask = masks[0] & masks[1] & masks[2]
            elif num_tasks == 2:
                mask = masks[0] & masks[1]

            # Use the mask to filter predicted_labels and labels
            filtered_preds = predicted_labels[mask]
            filtered_labels = labels[mask]
            
          
            if filtered_preds.nelement() == 0:
                accuracy_task[task,i] = 0.0  # or some default value
            else:
            
                accuracy_task[task,i] = np.nanmean((filtered_preds == filtered_labels).float())

    accuracy = np.nanmean(accuracy_task,axis=1)
    
   
    return accuracy,accuracy_task


def get_task_activity(net,tasks):
   
    dt = 100
    kwargs = {'dt': dt}
    seq_len = 100
    num_tasks = len(tasks)
    num_trial = 100
    cumsumtasks = 0
    activity_dict = {}
    trial_truth = np.zeros([num_tasks*num_trial])
    count = -1
   
    #get total number of input neurons
    s = net.state_dict()
    sh = s['rnn.input2h.weight'].shape
    total_in_size = sh[1]
    #initialize per task input size
    in_size = 0

    before = 0
    after = total_in_size
   
    for t in range(num_tasks):

        #initialize the task
        dataset = ngym.Dataset(tasks[t], env_kwargs=kwargs, batch_size=16,
                           seq_len=seq_len)
        env = dataset.env
       
        i1 = env.observation_space.shape[0]
        #before equals minus previous input size, after is total size minus previous task(s) input size(s)
        before +=in_size
        after -= i1
        in_size = i1
           
           

        env.reset(no_step=True)
        perf = 0
       
        for i in range(num_trial):
            count += 1
            env.new_trial()
            ob, gt = env.ob, env.gt
            inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float)
           
           
           
           
                    #translate inputs for multitask (each task gets its own set of nodes)

            inputs = np.pad(inputs,((0,0),(0,0),(before,after)))

            inputs = torch.from_numpy(inputs)
           
            if count == 0:
                all_inputs = inputs
            else:
                all_inputs = np.concatenate((all_inputs,inputs),axis = 0)
           
        
            if count == 0:
                    task_indices = np.ones(inputs.shape[0])
            else:
                    t_num = t+1
                    task_indices =  np.concatenate((task_indices,t_num*np.ones(inputs.shape[0])),axis = 0)
                    
                    
            action_pred, rnn_activity = net(inputs)
            rnn_activity = rnn_activity[:, 0, :].detach().numpy()
            activity_dict[count] = rnn_activity
           
            if count == 0:
                activity = activity_dict[count]
            else:  
                activity = np.concatenate((activity, activity_dict[count]), axis=0)
           
            trial_infos = env.trial
           
            #change trial correct output number for multiple tasks
           
            truth = np.array(trial_infos['ground_truth'])
            #some tasks say 'match' instead of giving a number, translate to number here:
            if truth == 'non-match':
                truth = 1
            elif truth == 'match':
                truth = 0
               
            tt = truth + cumsumtasks
            trial_truth[count] = tt+1
            #trial_truth = []
        #cumsumtasks = np.max(trial_truth)

    # Concatenate activity for PCA

   
    return activity, activity_dict, trial_truth, all_inputs, task_indices


def get_flow(model,activity, input_sequence,pca_result,nbins,seq_length,create_movie,plot_it):
    # Create flow diagram to see transient dynamics

    pca = PCA(n_components=2)
    pca.fit(activity)
    #Create a grid in the 2D PCA space
    x_vals = np.linspace(min(pca_result[:, 0])-2, max(pca_result[:, 0])+2, nbins)
    y_vals = np.linspace(min(pca_result[:, 1])-2, max(pca_result[:, 1])+2, nbins)

    grid_points_pca = np.array([[x, y] for x in x_vals for y in y_vals])



    # Project points back to original space
    grid_points_original = pca.inverse_transform(grid_points_pca)

    # Convert to tensor
    grid_points_original = torch.tensor(grid_points_original, dtype=torch.float)


    # ...

    # Initialize empty list to store the trajectories for each initial hidden state
    all_trajectories = []
    
    # Iterate through each initial hidden state in the grid
    for init_hidden in grid_points_original:
        # Initialize hidden state
        hidden = init_hidden.unsqueeze(0).unsqueeze(0)  # Add sequence and batch dimensions

        # Create input sequence (all zeros)
        #input_sequence = torch.zeros(sequence_length, 1, input_size)  # Single sequence

        # Store the evolution of this specific initial hidden state
        trajectories = []
        trajectories.append(hidden[:,0,:].squeeze().detach().numpy())
        # Evolve the hidden state
        for t in range(seq_length):
            hidden = model.rnn.recurrence(input_sequence, hidden)
            #net.rnn.recurrence
            #print(hidden.shape)
            trajectories.append(hidden[:,0,:].squeeze().detach().numpy())
        trajectories = np.array(trajectories)
        #print(trajectories.shape)
        # Transform the trajectory to PCA space for easy visualization
        trajectories_pca = pca.transform(trajectories)

        # Save this trajectory
        all_trajectories.append(trajectories_pca)

    # Convert to numpy array for easier indexing
    all_trajectories = np.array(all_trajectories)
    
    if plot_it == "yes":
        # Plotting in PCA space
        fig, ax = plt.subplots()
        
      

        

         # Choose some time steps to plot
        #time_steps_to_plot = range(0, seq_length - 1, 5)  # Until sequence_length - 1 because we'll look at next steps

        t = 0
        #for t in time_steps_to_plot:
        x_vals = all_trajectories[:, t, 0]  # First principal component at time t
        y_vals = all_trajectories[:, t, 1]  # Second principal component at time t
        
        # Create the scatter plot
        x = np.squeeze(all_trajectories[:,0,0])
        y = np.squeeze(all_trajectories[:,0,1])

        sc = ax.scatter(x, y, s=np.zeros(625), c=np.zeros(625), cmap='Greys', alpha=0.5)
        plt.colorbar(sc)
        # Compute differences to plot as vectors
        dx = all_trajectories[:, t + 1, 0] - x_vals
        dy = all_trajectories[:, t + 1, 1] - y_vals

        ax.quiver(x_vals, y_vals, dx, dy, angles='xy') #, scale_units='xy', scale=1

        ax.legend()
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('RNN Hidden State Flow Diagram in PCA Space (Quiver Plot)')
        plt.show()
    
    
    if create_movie == "yes":
        from matplotlib.animation import FuncAnimation

        # ...

        # Your existing code to generate all_trajectories goes here

        # ...

        fig, ax = plt.subplots()
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('RNN Hidden State Flow Diagram in PCA Space (Quiver Plot)')

        # Initialize a quiver plot
        Q = ax.quiver(x_vals, y_vals, dx, dy, angles='xy', label=f'Time {t}') #, scale_units='xy', scale=1
        
        all_quivers = []
        
        def init():
            # Initialize empty quiver data
            Q.set_UVC(x_vals, y_vals)
            Q.set_offsets([])
            return Q,

        def update(t):
            # Set quiver data for frame t
            x_vals = all_trajectories[:, t, 0]
            y_vals = all_trajectories[:, t, 1]
            dx = all_trajectories[:, t + 1, 0] - x_vals
            dy = all_trajectories[:, t + 1, 1] - y_vals
            Q.set_UVC(dx, dy)
            Q.set_offsets(np.column_stack((x_vals, y_vals)))
            return Q, 
        
        '''def update(t):
            global all_trajectories  # Since we're modifying it
            global all_quivers  # Declare it as global so that we can modify it

            # Extract current positions
            x_vals = all_trajectories[:, t, 0]
            y_vals = all_trajectories[:, t, 1]

            # Compute differences to plot as vectors
            dx = all_trajectories[:, t + 1, 0] - x_vals
            dy = all_trajectories[:, t + 1, 1] - y_vals

            all_quivers.append((x_vals.copy(), y_vals.copy(), dx.copy(), dy.copy()))

            # Concatenate all quivers to plot them all at the current frame
            all_x_vals = np.concatenate([q[0] for q in all_quivers])
            all_y_vals = np.concatenate([q[1] for q in all_quivers])
            all_dx = np.concatenate([q[2] for q in all_quivers])
            all_dy = np.concatenate([q[3] for q in all_quivers])

            # Clear the existing quivers
            ax.clear()

            # Create a new quiver plot with the updated positions and vectors
            Q = ax.quiver(all_x_vals, all_y_vals, all_dx, all_dy, angles='xy', scale_units='xy', scale=1)

            return Q,'''

        # Create animation
        ani = FuncAnimation(fig, update, frames=range(0, seq_length - 1), init_func=init, blit=True)

        # To view the animation inline (may not work in all environments)
        from IPython.display import HTML
        HTML(ani.to_jshtml())

        # Or save it
        ani.save('rnn_flow.mp4', writer='ffmpeg')

        plt.show()

    
    return all_trajectories


def lesion_rnn(model,start_end):
    """
    Lesion an RNN model at the given neuron index.

    Parameters:
    - model (torch.nn.Module): The PyTorch RNN model to lesion.
    - neuron_index (int): The index of the neuron to lesion.

    Returns:
    - torch.nn.Module: A new PyTorch RNN model with the specified neuron lesioned.
    """
    
    it = model.rnn.input2h.weight.detach().numpy()
    ot = model.fc.weight.detach().numpy()
    input_size = it.shape[1]
    output_size = ot.shape[0]
    hidden_size = 100
    dt = 100

    # Clone the original model
    
    lesioned_model = RNNNet(input_size=input_size, hidden_size=hidden_size,
             output_size=output_size, dt=dt)
    mask = np.ones((100,100))
    mask = torch.from_numpy(mask)
    apply = prune.custom_from_mask(lesioned_model.rnn.h2h, name = "weight", mask = mask)
    
    # Load the original weights into this new instance
    lesioned_model.load_state_dict(model.state_dict())

    # Lesion the weights for the neuron in the hidden-to-hidden transition
    lesioned_model.rnn.h2h.weight_orig.data[start_end[0]:start_end[1], start_end[0]:start_end[1]] = 0.0
   
    #plt.imshow(lesioned_model.rnn.h2h.weight_orig.data)
    #plt.show()
    # Lesion the biases for the neuron in the hidden layer
    lesioned_model.rnn.h2h.bias.data[start_end[0]:start_end[1]] = 0.0


    return lesioned_model

def lesion_rnn_mask(model,mask):
    """
    Lesion an RNN model at the given neuron index.

    Parameters:
    - model (torch.nn.Module): The PyTorch RNN model to lesion.
    - neuron_index (int): The index of the neuron to lesion.

    Returns:
    - torch.nn.Module: A new PyTorch RNN model with the specified neuron lesioned.
    """
    
    it = model.rnn.input2h.weight.detach().numpy()
    ot = model.fc.weight.detach().numpy()
    input_size = it.shape[1]
    output_size = ot.shape[0]
    hidden_size = 100
    dt = 100

    # Clone the original model
    
    lesioned_model = RNNNet(input_size=input_size, hidden_size=hidden_size,
             output_size=output_size, dt=dt)
 
    # Load the original weights into this new instance
    lesioned_model.load_state_dict(model.state_dict())

    # Lesion the weights for the neuron in the hidden-to-hidden transition
    lesioned_model.rnn.h2h.weight.data[mask] = 0.0
   


    return lesioned_model

from scipy.optimize import linear_sum_assignment
import networkx as nx





def SC_modules_thalamic_bias(off_block, thalamic_bias,tasks):
     #create modular pruning mask with stochastic block model
    sizes = [33,33,34]
    #off_block = 0.1
    probs = [[1, off_block,off_block],[off_block, 1, off_block],[off_block,off_block, 1]]
    g = nx.stochastic_block_model(sizes,probs,directed=True)
    gg = nx.to_numpy_array(g)

    
    np.fill_diagonal(gg, 1)

    #plt.imshow(gg)
    #plt.colorbar()
   # lt.show()


    steps = 500
    mask = gg
    randomize_task_order = 1
    lr = 0.01
    hidden_size = 100
    batch_size = 16
    num_runs = 1
    

    IW = np.random.randn(100,12)

    #thalamic_bias = 3

    IW[0:33,0:3] += thalamic_bias
    IW[33:66,3:6] += thalamic_bias
    IW[66:100,6:12] += thalamic_bias

    #plt.imshow(IW)
    #plt.colorbar()
    #plt.show()

    input_weights = IW
    
    net_modules, loss_trajectory = train_multitask2(tasks,steps,mask,lr,randomize_task_order, input_weights)

    A = np.zeros((3,3))

    start_end = [0, 33]
    lesioned_model = lesion_rnn(net_modules,start_end)
    A[0,:],accuracy_task = get_accuracy_per_task(lesioned_model,tasks,hidden_size,batch_size,num_runs)

    start_end = [33, 66]
    lesioned_model = lesion_rnn(net_modules,start_end)
    A[1,:],accuracy_task = get_accuracy_per_task(lesioned_model,tasks,hidden_size,batch_size,num_runs)

    start_end = [66, 100]
    lesioned_model = lesion_rnn(net_modules,start_end)
    A[2,:],accuracy_task = get_accuracy_per_task(lesioned_model,tasks,hidden_size,batch_size,num_runs)




    print("A",A)
    D = np.mean(A.diagonal())
    i, j = np.indices(A.shape)
    off_diagonal = A[i != j]
    ND = np.mean(off_diagonal)

    specialization_quotient = ND-D

    print("SQ",specialization_quotient)
    
    
    row_indices, col_indices = linear_sum_assignment(A)
    optimal_A = A[:, col_indices]
    
    
    D = np.mean(optimal_A.diagonal())
    i, j = np.indices(optimal_A.shape)
    off_diagonal = optimal_A[i != j]
    ND = np.mean(off_diagonal)

    optimal_specialization_quotient = ND-D
    
    print("OSQ",optimal_specialization_quotient)
    
    return net_modules, specialization_quotient, optimal_specialization_quotient





# Compute PCA and visualize
from sklearn.decomposition import PCA



def get_phase_distance(net, tasks):
    activity, activity_dict, trial_truth, all_inputs, task_indices = get_task_activity(net,tasks)

    pca = PCA(n_components=2)
    pca.fit(activity)
    # print('Shape of the projected activity: (Time points, PCs): ', activity_pc.shape)


    activity_pc = pca.transform(activity)

    dt = 100
    kwargs = {'dt': dt}
    seq_len = 100
    num_tasks = len(tasks)
    i1 = np.zeros([num_tasks])
    o1 = np.zeros([num_tasks])
    
    for task in range(num_tasks):
      
        dataset = ngym.Dataset(tasks[task], env_kwargs=kwargs, batch_size=16,
                       seq_len=seq_len)
       
                   #get input and output sizes for different tasks
        env = dataset.env
        i1[task] = env.observation_space.shape[0]
        try:
            o1[task] = env.action_space.n
        except:
            o1[task] = env.action_space.shape[0]



    nbins = 25
    input_sequence = {}
    #task 1 fixation
    for task in range(num_tasks):
        if task > 0:
            tiler = np.zeros([int(np.sum(i1))])
            #assuming the fixation is the first input for each of them
            tiler[int(i1[task-1])+1] = 1
            input_sequence[task] = np.tile(tiler, (2, 1))
            input_sequence[task] = torch.tensor(input_sequence[task], dtype=torch.float32)
        else:
            tiler = np.zeros([int(np.sum(i1))])
            #assuming the fixation is the first input for each of them
            tiler[0] = 1
            input_sequence[task] = np.tile(tiler, (2, 1))
            input_sequence[task] = torch.tensor(input_sequence[task], dtype=torch.float32)

    seq_length = 10

    create_movie = "no"
    plot_it = "no"
    all_quivers = []
    all_trajectories = {}
    for i in range(num_tasks):
        all_trajectories[i] = get_flow(net,activity, input_sequence[i],activity_pc,nbins,seq_length,create_movie,plot_it)
        


    def euclidean_distance(p, q):
        return np.linalg.norm(np.array(p) - np.array(q))

    S = all_trajectories[0].shape
    ED = np.zeros(S[0])
    phase_space_dist = np.zeros((num_tasks,num_tasks))
    for task1 in range(num_tasks):
        for task2 in range(num_tasks):
            for i in range(S[0]):
                ED[i] = euclidean_distance(all_trajectories[task1][i,1,:], all_trajectories[task2][i,1,:])
                
                phase_space_dist[task1,task2] = np.sum(ED)

    return phase_space_dist



