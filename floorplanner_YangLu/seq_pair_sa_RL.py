#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: seq_pair_obs.py
Created on: oct 30 3:15 PM 2019
@author: yanglu
"""
import csv
import numpy as np
from simanneal import Annealer
import random
import matplotlib.pyplot as plt
from floorplanner_YangLu.seq_pair_obs import seq_pair_blockage
from floorplanner_YangLu.seq_operators import *

#TODO: for efficient space search, we need to limit the range that 
# move func operate on. one method is only swap or rotate block i and j
# which is close in physical space. in order to do that,  we can define
# certain uniform grid bins and assign each block to certain bins according 
# to the block's coordindates.  we looking for neighboring blocks, simply search
# in neighboring bins, in the context of force directed method, simply search 
# the bins in the direction of the force.

## another way to do the directed move is to use 
## move --> legalize iteration.  the move is stochastically
## directed by the force field.  legalization make sure on overlap happens

class SeqPairSA(Annealer) :
  """
  optimize seq-pair by simulated annealing alg
  """

  def __init__(self, state, sp:seq_pair_blockage, steps) :
    self.sp = sp
    self.num_block = sp.num_block
    self.state = state
    self.set_schedule(self.auto(minutes=0.2))
    self.steps = steps    
    super(SeqPairSA, self).__init__(state)


  def move(self) :
    
    if random.random() < 0.1:
      self.move_global()
    else :  
      #self.move_local(self.num_block/10) 
      self.move_local_rev(1)
     
    #self.move_local(self.num_block/10) 
    #self.move_local_rev(1) 
    #self.move_global()   
   


  def move_global(self) :
    dice = random.random()

    if dice < 0.66 : #0.66
      pool = list(range(self.num_block))
      i,j = random.sample(pool, 2)
      pos_seq = self.state[0:self.num_block]
      neg_seq = self.state[self.num_block:2*self.num_block]

      if dice < 0.33 :
        seq_pair_rotate(pos_seq, neg_seq, i, j, 0)
      else :
        seq_pair_swap(pos_seq, neg_seq, i, j)

      self.state[0:self.num_block] = pos_seq
      self.state[self.num_block:2*self.num_block] = neg_seq  

    else : #################################
      # flip shape gene
      i = random.randint(0, self.num_block-1)
      self.state[2*self.num_block+i] = 1 - self.state[2*self.num_block+i]


  def move_local(self, sigma) :
      dice = random.random()
      i = random.randint(0, self.num_block-1)
      di = max(int(random.gauss(0, sigma)), 1)
      
      if random.random() < 0.5 :
        j = i + di
      else :
        j = i - di

      j = max(min(j, self.num_block-1), 0)
      
      if i == j : 
        return

      pos_seq = self.state[0:self.num_block]
      neg_seq = self.state[self.num_block:2*self.num_block]

      if dice < 0.5 :
        seq_pair_rotate(pos_seq, neg_seq, i, j, 0)
      else :
        seq_pair_swap(pos_seq, neg_seq, i, j)

      self.state[0:self.num_block] = pos_seq
      self.state[self.num_block:2*self.num_block] = neg_seq  


  def move_local_rev(self, sigma) :
    i = random.randint(0, self.num_block-1) # selecting block to move
    di = max(int(random.gauss(0, sigma)), 1)
    j = max(min(i+di, self.num_block-1), 0) #selecting block to replace with
    if i == j :
      return 

    dice = random.random()
    if dice < 0.5 :  # move ith block in  seq1
      seq = self.state[0:self.num_block]
    else :  # move ith block in seq2
      seq = self.state[self.num_block:2*self.num_block]
    
    x = seq[i]
    if j > i :
     seq[i:j] = seq[i+1:j+1]
     seq[j] = x
    else :
     seq[j+1:i+1] = seq[j:i]
     seq[j] = x 

    if dice < 0.5 :
      self.state[0:self.num_block] = seq
    else :
      self.state[self.num_block:2*self.num_block] = seq


  def energy(self) : #the objective function to be minimized; temperature is for getting out of global minima
    seq1  =  self.state[:self.num_block]
    seq2  =  self.state[self.num_block:2 * self.num_block]
    onehot_random =  self.state[2*self.num_block:]
    # self.sp.pack(seq1, seq2, shape) #get feasible soln.
    self.sp.pack ( seq1, seq2, [])
    return self.sp.evaluate()


   
      
if __name__ == '__main__':
  seed = 25
  random.seed(seed)
  np.random.seed(seed)

  blks_info_ami49 = []
  with open('ami49.txt') as csvfile :
    reader = csv.reader(csvfile, delimiter=' ')
    for row_raw in reader :
      name = row_raw[0]
      w    = int(row_raw[1])
      h    = int(row_raw[4])
      blks_info_ami49.append((name, (None, None), (w, h)))

  print(blks_info_ami49)  

  # let's fix module M001
  blks_info_ami49[0] = ('M001', (0, 630), (3234, 1708))
  # let's fix module M004
  blks_info_ami49[3] = ('M004', (630, 5040), (3080, 1610))
  # let's fix module M006
  blks_info_ami49[5] = ('M006', (4774, 2548), (882, 1862))
  netlist_info = []
  sp = seq_pair_blockage(blks_info_ami49, netlist_info, do_adapt = True)

  seq1 = random.sample(range(sp.num_block), sp.num_block)
  seq2 = random.sample(range(sp.num_block), sp.num_block)

  random_block = random.sample ( list ( range ( sp.num_block ) ),
                                      1 )  # Randomly pick index of first block to be swapped
  onehot_random = [0] * sp.num_block  # generate one hot encoding of randomly chosen block
  onehot_random[random_block[0]] = 1
  # shape = [random.randint(0,1) for _ in range(sp.num_block)]
  init_state = seq1 + seq2 + onehot_random
  sa = SeqPairSA(init_state, sp)

  sa.set_schedule({'tmax': 75000, 'tmin': 2.5, 'steps': 1000, 'updates': 10}) ###############################

  sa.copy_strategy = "slice"

  state, e = sa.anneal()

  print('final energy = ', e)

  seq1_res = state[:sp.num_block]
  seq2_res = state[sp.num_block:2*sp.num_block]


  sp.pack(seq1_res,seq2_res, [])

  sp.plot()

  plt.show()
