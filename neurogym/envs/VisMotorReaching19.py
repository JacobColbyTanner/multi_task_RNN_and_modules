#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import neurogym as ngym
from neurogym import spaces

class VisMotorReaching19(ngym.TrialEnv):
    metadata = {
        'paper_link': 'n/a',
        'paper_name': 'Sine Wave Future State Prediction',
        'tags': ['time series', 'prediction']
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

            
            

        
        
        name = {'fixation': 0, 'visual': [1,2], 'motor': [3,4]}

        # Observation space: 1 for fixation, 2,3 for location of visual object, 4,5 for current motor position
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32, name=name)
        # Action space: 1 output to predict the future state
        name = {'decision': [1,2]}
        #first is fixation, next two are motor coordinates
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32, name=name)
        #self.action_space = spaces.Discrete(3, name=name)
        
    def _new_trial(self, **kwargs):
        # Initialize variables
        self.timing = {
            'fixation': 0,
            'visual': np.random.randint(400,high=2000),
            'motor': np.random.randint(400,high=2000),
            'decision': 200}
        
        trial = {'ground_truth': 0}
        trial.update(kwargs)

        target_location = self.np_random.uniform(-1, 1, size=(2,))
        limb_position = self.np_random.uniform(-1, 1, size=(2,))
        
        # Add periods
        self.add_period(['fixation', 'visual', 'motor', 'decision'])
        
        # Observations
        self.add_ob(1, period=['fixation', 'visual', 'motor'], where='fixation')
        
        self.add_ob(target_location, 'visual', where='visual')
        self.add_ob(limb_position, 'motor', where='motor')
        
        truth = target_location-limb_position
        gtt = np.zeros(3)
        gtt[1:3] = truth
        self.set_groundtruth(gtt, 'decision')
        
        return trial

    def _step(self, action):
        # Initialize reward and info
        reward = 0
        info = {'new_trial': False, 'gt': self.gt_now}
        
        # Check for abort
        if self.in_period('fixation') and self.ob_now[0] == 0:
            reward += self.rewards['abort']
            info['new_trial'] = True
        
        
        # Check for decision
        if self.in_period('decision'):
            info['new_trial'] = True

            if np.isclose(action, self.gt_now, atol=0.1).all():
                reward += self.rewards['correct']
            else:
                reward += self.rewards['fail']

        return self.ob_now, reward, False, info



# In[1]:




