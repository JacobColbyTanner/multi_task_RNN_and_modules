
import numpy
import functions
import neurogym 
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import time
#import scipy.io as sio

from gym.envs.registration import register

register(id = 'gonogo_variable_delay-v0',entry_point= "neurogym.envs.gonogo_variable_delay:gonogo_variable_delay")

import gym

env = gym.make('gonogo_variable_delay-v0')  # Replace 'YourEnvName-v0' with the ID you used during registration


from gym.envs.registration import register

register(id = 'OrientedBar-v0',entry_point= "neurogym.envs.OrientedBar7:OrientedBar7")



env = gym.make('OrientedBar-v0')  # Replace 'YourEnvName-v0' with the ID you used during registration




from gym.envs.registration import register

register(id = 'VisMotorReaching-v0',entry_point= "neurogym.envs.VisMotorReaching19:VisMotorReaching19")



env = gym.make('VisMotorReaching-v0')  # Replace 'YourEnvName-v0' with the ID you used during registration



from gym.envs.registration import register

register(id = 'ObjectSequenceMemory-v0',entry_point= "neurogym.envs.ObjectSequenceMemory24:ObjectSequenceMemory24")



env = gym.make('ObjectSequenceMemory-v0')  # Replace 'YourEnvName-v0' with the ID you used during registration



tasks = ['PerceptualDecisionMaking-v0','gonogo_variable_delay-v0','ObjectSequenceMemory-v0'] #'GoNogo-v0',






import time
import scipy.io as sio

dt = 100
kwargs = {'dt': dt}
seq_len = 100



off_block = [0,0.125,0.25,0.5,1]
thalamic_bias = [0,1,2,3]

ob_S = len(off_block)
tb_S = len(thalamic_bias)
specialization_quotient = np.zeros((100,ob_S,tb_S))
optimal_specialization_quotient = np.zeros((100,ob_S,tb_S))
phase_space_distances_all = np.zeros((100,ob_S,tb_S))

for iter in range(100):
    print("-------------------------------iteration: ",iter,flush=True)
    
    np.save("data/SQ.npy",specialization_quotient)
    np.save("data/OSQ.npy",optimal_specialization_quotient)
    np.save("data/phase_distance.npy",phase_space_distances_all)
    for ob in range(ob_S):
        for tb in range(tb_S):
            start_time = time.time()

            net_modules,specialization_quotient[iter,ob,tb], optimal_specialization_quotient[iter,ob,tb] = functions.SC_modules_thalamic_bias(off_block[ob], thalamic_bias[tb],tasks)

            phase_space_dist = functions.get_phase_distance(net_modules,tasks)

            dd = np.triu(phase_space_dist, k=1)

            phase_space_distances_all[iter,ob,tb] = np.sum(dd)

            end_time = time.time()
            elapsed_time = end_time - start_time



