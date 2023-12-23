#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import neurogym as ngym
from neurogym import spaces


class OrientedBar7(ngym.TrialEnv):


    def __init__(self, dt=100, rewards=None, timing=None):
        super().__init__(dt=dt)


        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': 100,
            'stimulus': 2000,
            'decision': 100}
        if timing:
            self.timing.update(timing)

        self.abort = False

        #self.theta = np.linspace(0, 2*np.pi, dim_ring+1)[:-1]
        #self.choices = np.arange(dim_ring)

        #create 4 x 4 image space
        name = {'fixation': 0, 'stimulus': range(1, 17)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1+16,), dtype=np.float32, name=name)
        #respond to 4 different orientations
        name = {'fixation': 0, 'choice': range(1, 5+1)}
        self.action_space = spaces.Discrete(5, name=name)

    def _new_trial(self, **kwargs):

        # Trial info
        trial = {
            'ground_truth': np.random.randint(1,high=5) #random int in range 1 to 4
            }
        trial.update(kwargs)

        
        ground_truth = trial['ground_truth']
        

        # Periods
        self.add_period(['fixation', 'stimulus', 'decision'])

        # Observations
        self.add_ob(1, period=['fixation', 'stimulus'], where='fixation')
        
        
        
        #define stim based on orientation
        
        if ground_truth == 1:
            #horizontal
            #four options
            select = np.random.randint(1,high=5)
            #one
            if select == 1:
                image = np.zeros(16)
                image[0:4] = 1
            
            #two
            if select == 2:
                image = np.zeros(16)
                image[4:8] = 1
            
            #three 
            if select == 3:
                image = np.zeros(16)
                image[8:12] = 1
            
            #four
            if select == 4:
                image = np.zeros(16)
                image[12:16] = 1
            
            
        elif ground_truth == 2:
            #vertical
            #four options
            select = np.random.randint(1,high=5)
            #one
            if select == 1:
                image = np.zeros(16)
                image[[0,4,8,12]] = 1
            
            #two
            if select == 2:
                image = np.zeros(16)
                image[[1,5,9,13]] = 1
            
            #three 
            if select == 3:
                image = np.zeros(16)
                image[[2,6,10,14]] = 1
            
            #four
            if select == 4:
                image = np.zeros(16)
                image[[3,7,11,15]] = 1
            
            
        elif ground_truth == 3:
            #left up diagonal
            #three options
            select = np.random.randint(1,high=4)
            #one
            if select == 1:
                image = np.zeros(16)
                image[[0,5,10,15]] = 1
            
            #two
            if select == 2:
                image = np.zeros(16)
                image[[1,6,11]] = 1
            
            #three 
            if select == 3:
                image = np.zeros(16)
                image[[4,9,14]] = 1
        
        elif ground_truth == 4:
            #right up diagonal
            #three options
            select = np.random.randint(1,high=4)
            #one
            if select == 1:
                image = np.zeros(16)
                image[[3,6,9,12]] = 1
            
            #two
            if select == 2:
                image = np.zeros(16)
                image[[2,5,8]] = 1
            
            #three 
            if select == 3:
                image = np.zeros(16)
                image[[7,10,13]] = 1
            
        stim = image
        
        
        self.add_ob(stim, 'stimulus', where='stimulus')
        

        # Ground truth
        self.set_groundtruth(ground_truth-1, period='decision', where='choice')

        return trial

    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now
        # observations
        if self.in_period('fixation'):
            if action != 0:  # action = 0 means fixating
                new_trial = self.abort
                reward += self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward += self.rewards['correct']
                    self.performance = 1
                else:
                    reward += self.rewards['fail']

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}

