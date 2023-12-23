#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import neurogym as ngym
from neurogym import spaces

class ObjectSequenceMemory24(ngym.TrialEnv):
    metadata = {
        'paper_link': 'n/a',
        'paper_name': 'The importance of mixed selectivity in complex cognitive tasks',
        'tags': ['mixed_selectivity', 'cognition']
    }

    def __init__(self, dt=100, rewards=None, timing=None):
        super().__init__(dt=dt)

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': 100,
            'visual': 2000,
            'motor': 2000,
            'decision': 200}
        
        
            
        if timing:
            self.timing.update(timing)

            
            

        
        
        name = {'fixation': 0, 'rule_input': [1], 'objects': [2,3,4,5]}

        # Observation space: 1 for fixation, 
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32, name=name)
        # Action space: 
        name = {'decision': range(0, 6)}
        #first is fixation, second is 'decision' on recognition trials, third through sixth are object types
        #self.action_space = spaces.Box(-np.inf, np.inf, shape=(6), dtype=np.float32, name=name)

        self.action_space = spaces.Discrete(6, name=name)
        
    def _new_trial(self, **kwargs):
        # Initialize variables
        
        self.trial_type = np.random.randint(0,high=2)
        
        if self.trial_type == 0: #recognition trial
            self.timing = {
                'fixation':0,
                'first_delay': 1000,
                'first_cue': 500,
                'second_delay': 1000,
                'second_cue': 500,
                'third_delay': 1000,
                'first_delay2': 1000,
                'first_cue2': 500,
                'second_delay2': 1000,
                'second_cue2': 500,
                'third_delay2': 1000,
                'decision': 200}

            trial = {'ground_truth': 0}
            trial.update(kwargs)


            # Add periods
            self.add_period(['fixation','first_delay','first_cue','second_delay','second_cue','third_delay', 'first_delay2','first_cue2','second_delay2','second_cue2','third_delay2', 'decision'])

            # Observations #add trial fixation
            self.add_ob(1, period=['fixation','first_delay','first_cue','second_delay','second_cue','third_delay','first_delay2','first_cue2','second_delay2','second_cue2','third_delay2'], where='fixation')
            
            same_sequence = np.random.randint(0,high=2)
            
            #rule input
            self.add_ob(self.trial_type, period=['fixation','first_delay','first_cue','second_delay','second_cue','third_delay','first_delay2','first_cue2','second_delay2','second_cue2','third_delay2'], where='rule_input')
            
            if same_sequence == 0:
                obs = np.random.randint(0,high=4,size=2)

                object_select1 = np.zeros(4)
                object_select1[obs[0]] = 1
                object_select2 = np.zeros(4)
                object_select2[obs[1]] = 1
                
                
                
                #first sequence
                self.add_ob(object_select1, 'first_cue', where='objects')
                self.add_ob(object_select2, 'second_cue', where='objects')
                
                #second sequence
                self.add_ob(object_select1, 'first_cue2', where='objects')
                self.add_ob(object_select2, 'second_cue2', where='objects')
                
                

                
                self.set_groundtruth(1, 'decision')
                
            if same_sequence == 1:
                
                obs = np.random.randint(0,high=4,size=2)

                object_select1 = np.zeros(4)
                object_select1[obs[0]] = 1
                object_select2 = np.zeros(4)
                object_select2[obs[1]] = 1
                
                
                
                #first sequence
                self.add_ob(object_select1, 'first_cue', where='objects')
                self.add_ob(object_select2, 'second_cue', where='objects')
                
                obs2 = np.random.randint(0,high=4,size=2)
                
                while np.sum(obs == obs2) == 2: #make sure they are different
                    obs2 = np.random.randint(0,high=4,size=2)

                object_select1 = np.zeros(4)
                object_select1[obs2[0]] = 1
                object_select2 = np.zeros(4)
                object_select2[obs2[1]] = 1

                #second sequence
                self.add_ob(object_select1, 'first_cue2', where='objects')
                self.add_ob(object_select2, 'second_cue2', where='objects')
                
                

                
                self.set_groundtruth(0, 'decision')
            
            
            
        elif self.trial_type == 1:
            
            self.timing = {
                'fixation': 0,
                'first_delay': 1000,
                'first_cue': 500,
                'second_delay': 1000,
                'second_cue': 500,
                'third_delay': 1000,
                'decision1': 100,
                'decision2': 100}

            trial = {'ground_truth': 0}
            trial.update(kwargs)

            # Add periods
            self.add_period(['fixation','first_delay','first_cue','second_delay','second_cue','third_delay', 'decision1', 'decision2'])

            # Observations #add trial fixation
            self.add_ob(1, period=['fixation','first_delay','first_cue','second_delay','second_cue','third_delay'], where='fixation')
            #rule input
            self.add_ob(self.trial_type, period=['fixation','first_delay','first_cue','second_delay','second_cue','third_delay'], where='rule_input')
            
            
            obs = np.random.randint(0,high=4,size=2)

            object_select1 = np.zeros(4)
            object_select1[obs[0]] = 1
            object_select2 = np.zeros(4)
            object_select2[obs[1]] = 1
            

            #first sequence
            self.add_ob(object_select1, 'first_cue', where='objects')
            self.add_ob(object_select2, 'second_cue', where='objects')
            
            
            self.set_groundtruth(obs[0]+2, 'decision1')
            self.set_groundtruth(obs[1]+2, 'decision2')
        
        return trial

    def _step(self, action):
        # Initialize reward and info
        reward = 0
        info = {'new_trial': False, 'gt': self.gt_now}
        
        # Check for abort
        if self.in_period('fixation') and self.ob_now[0] == 0:
            reward += self.rewards['abort']
            info['new_trial'] = True
        
        info['trial_type'] = self.trial_type
        

        return self.ob_now, reward, False, info


