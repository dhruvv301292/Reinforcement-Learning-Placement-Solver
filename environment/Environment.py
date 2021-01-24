import gym
from gym import spaces
import numpy as np
import random


# Edit: haiguang 08/17/2020
from floorplanner_YangLu.seq_operators import *
from floorplanner_YangLu.seq_pair_obs import *
# from floorplanner_YangLu.seq_pair_fast_cpp import *



class Custom(gym.Env):
    def __init__(self, block_info):

        self.blocks_info = block_info
        self.num_block = len(self.blocks_info)
        self.action_space = spaces.Discrete(self.num_block)  # selecting 1 block from all available blocks.
        # Edit: haiguang 08/17/2020 shape size
        self.observation_space = spaces.Box(low=0, high=self.num_block, shape=(3*self.num_block,), dtype=np.int64)  # array of both the sequences + array of 1-hot encoded random block number
        self.sp = seq_pair_blockage(self.blocks_info, [], do_adapt=True)  # sp block now created in init instead of step
        self.seq1 = random.sample(range(self.num_block), self.num_block)  # seq1 and seq2 defined in init
        self.seq2 = random.sample(range(self.num_block), self.num_block)
        
        #edit: dhruv 8/19 start        
        self.random_block = random.sample(list(range(self.num_block)), 1)  # Randomly pick index of first block to be swapped
        onehot_random = [0] * self.num_block  # generate one hot encoding of randomly chosen block
        onehot_random[self.random_block[0]] = 1
        #edit: dhruv 8/19 end
        
        self.obs = self.seq1 + self.seq2 + onehot_random  # added onehot encoding of random block to observed state 
        self.reward = 0
        self.obs_init = self.seq1 + self.seq2
        self.max_step = 100
        self.count_step = 0
        self.is_terminal = False

    def perturb(self, seq1, seq2, actor_output):  # actor_output: Index to be swapped
        
        pos_seq = seq1
        neg_seq = seq2
        # Edit: haiguang 08/17/2020 i[0]
        seq_pair_swap ( pos_seq, neg_seq, self.random_block[0], actor_output )
        seq1 = pos_seq
        seq2 = neg_seq
        return seq1, seq2

    def evaluate(self, seq1, seq2):

        self.sp.pack ( seq1, seq2, [] )
        return self.sp.evaluate ()

    def step(self, actor_output):  # Removed obs from passed parameters

        self.seq1 = self.obs[:self.num_block]
        self.seq2 = self.obs[self.num_block:2 * self.num_block]

        old_area = self.evaluate(self.seq1, self.seq2)        
        self.seq1, self.seq2 = self.perturb(self.seq1, self.seq2, actor_output)  # New seq pair
        new_area = self.evaluate(self.seq1, self.seq2)
        
        self.count_step = self.count_step + 1
        if self.count_step >= self.max_step: 
            self.is_terminal = True
        # if new_area < old_area:
        #     self.reward = old_area - new_area
        # else:
        #     self.reward = 0  # or -1?

        self.reward = old_area - new_area        
            
        #edit: dhruv 8/19 start        
        self.random_block = random.sample(list(range(self.num_block)), 1)  # Randomly pick index of first block to be swapped
        onehot_random = [0] * self.num_block  # generate one hot encoding of randomly chosen block
        onehot_random[self.random_block[0]] = 1
        #edit: dhruv 8/19 end
        
        self.obs = self.seq1 + self.seq2 + onehot_random

        # print('self.reward',self.reward)

        return self.obs, self.reward, self.is_terminal # changed from next_state to obs

    def reset(self):
        self.reward = 0
        #edit: dhruv 8/19 start        
        self.random_block = random.sample(list(range(self.num_block)), 1)  # Randomly pick index of first block to be swapped
        onehot_random = [0] * self.num_block  # generate one hot encoding of randomly chosen block
        onehot_random[self.random_block[0]] = 1
        self.obs = self.obs_init + onehot_random
        #edit: dhruv 8/19 end
        
        return self.obs


    def render(self, mode='human', close=False):
        print("****************************")
        print(self.seq1, self.seq2)
        print(self.reward)
        print("*****************************" )
