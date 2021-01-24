#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: seq_pair_obs.py
Created on: oct 30 3:15 PM 2019
@author: yanglu
"""

from functools import cmp_to_key
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

TEST_COORD_CLAC = 1 # 0 for old routine, 1 for new routine

class seq_pair_blockage :
  """
  sequence pair with blockage
  ref: "VLSI/PCB PLACEMENT WITH OBSTACLES BASED ON SEQUENCE-PAIR"
  """

  def __init__(self, blocks_info, netlist_info, do_adapt = True) :
    self.blks_info = blocks_info
    self.netlist_info = netlist_info
    self.num_block = len(self.blks_info)
    self.coords =  np.zeros((self.num_block, 2))
    self.do_adapt = do_adapt
    
    self.is_fixed = np.zeros(self.num_block, dtype=bool) 

    self.blk_size = np.zeros((self.num_block, 2))

    # initial position
    self.blk_xy   = np.zeros((self.num_block, 2))

    self.w_area = 0.5
    self.w_wireLen = 0.5
    
    for i in range(self.num_block) :
      if self.blks_info[i][1][0] != None :
        self.is_fixed[i] = True

      self.blk_size[i][0], self.blk_size[i][1] = self.blks_info[i][2][0], self.blks_info[i][2][1]
      self.blk_xy[i]      = self.blks_info[i][1]   

    print(self.is_fixed)
    print(self.blk_size)

  def set_area_wireLen_weights(self, w_area, w_wireLen) :
    self.w_area = w_area
    self.w_wireLen = w_wireLen

  
  def set_shape(self, shape) :

    if len(shape) == 0 :
      return 
   
    for i in range(self.num_block) :
      if (not self.is_fixed[i]) and (shape[i]==1) : # flip
        self.blk_size[i][0], self.blk_size[i][1]= self.blks_info[i][2][1], self.blks_info[i][2][0]
      else :
        self.blk_size[i][0], self.blk_size[i][1]= self.blks_info[i][2][0], self.blks_info[i][2][1]        




  def evaluate(self) :
    """
    evaluate the quality of placement, by default
    return the area of the minmum bounding box
    """
    return self.w_area * self.calc_area() + self.w_wireLen * self.calc_wireLen()
    


  def calc_area(self) :
    INF_NEG, INF = -1e100, 1e00
    xmin, ymin, xmax, ymax = INF, INF, INF_NEG, INF_NEG

    for i in range(self.num_block):
      w = self.blk_size[i][0]
      h = self.blk_size[i][1]
      xl = self.coords[i][0]
      yl = self.coords[i][1]
      xmin = min(xmin, xl)
      ymin = min(ymin, yl)
      xmax = max(xmax, xl + w)
      ymax = max(ymax, yl + h)


    return (ymax - ymin) * (xmax - xmin) 


  def calc_wireLen(self) :
    cost = 0
    for info in self.netlist_info :
      indx1 = info[1][0]
      indx2 = info[1][1]
      x1,y1 = self.coords[indx1]
      w1,h1 = self.blk_size[indx1]
      x2,y2 = self.coords[indx2]
      w2,h2 = self.blk_size[indx2]

      cost += abs((x2+w2/2) -  (x1+w1/2)) + abs((y2+h2/2) -  (y1+h1/2)) 


    return cost
       


  

  def pack(self, seq1, seq2, shape) :
    self.coords =  np.zeros((self.num_block, 2))
    self.set_shape(shape)
    if self.do_adapt :
      return self.pack_with_adapt(seq1, seq2)
    else :
      return self.pack_no_adapt(seq1, seq2)


  def pack_no_adapt(self, seq1, seq2) :
    """
    no adaption
    """  
    self.seq1 = copy.deepcopy(seq1)
    self.seq2 = copy.deepcopy(seq2)
    self.indx_in_seq1 = self.calc_indx_in_seq1()
    self.calc_propped_coords(self.num_block-1)

    return self.seq1, self.seq2



  def pack_with_adapt(self, seq1, seq2) :  
    """
    deal with case with fixed modules
    return adpated sequence
    """
    self.seq1 = copy.deepcopy(seq1)

    self.indx_in_seq1 = self.calc_indx_in_seq1()
    # step 1 topologically sort seq2
    self.seq2 = self.topo_sort_seq2(seq2)



    # step 2, iterate
    for k in range(self.num_block) :
      #print('k = ', k)
      #print(self.seq1, self.seq2)
    
      if not self.is_fixed[self.seq2[k]] :  # free blk, calc coords

        if TEST_COORD_CLAC == 0 :
          self.calc_propped_coords(k)
        else :  
          self.calc_propped_coords_step(k)
        pk = self.seq2[k]  # placeable k
        x, y = self.coords[pk][0], self.coords[pk][1]
        w, h = self.blk_size[pk][0], self.blk_size[pk][1]
      else :
        self.step_2p4(k)
        continue
      
      # step 2.2
      q = None
      for i in range(k+1, self.num_block) :
        #xy_i = self.blks_info[self.seq2[i]][1]
        xy_i = self.blk_xy[self.seq2[i]]
        # found fixed module that dominate pk
        if self.is_fixed[self.seq2[i]] and  self.is_dominate(xy_i[0], xy_i[1], x, y, w, h) :
          q = i
          break

      if q == None :  # next iteration
        continue     

      # step 2.3, move q before k in the seq2
      self.move_elem_left(self.seq2, q, k)


      # step 2.4
      self.step_2p4(k)

    return self.seq1, self.seq2  
     

  
  def step_2p4(self,k):
    """
    utililty method 
    """
    pk = self.seq2[k]
    
    if not self.is_fixed[pk] : 
      return
    
    if TEST_COORD_CLAC == 0 :
      self.calc_propped_coords(k)
    else :  
      self.calc_propped_coords_step(k)
    
    x,  y = self.coords[pk][0], self.coords[pk][1]  # propped coords of fixed blk
    xa, ya = self.blk_xy[pk][0], self.blk_xy[pk][1]
    k1 = self.indx_in_seq1[pk] 
    
    if x > xa :
       while True :
        self.seq1[k1], self.seq1[k1-1] = self.seq1[k1-1], self.seq1[k1] 
        self.indx_in_seq1[self.seq1[k1]],self.indx_in_seq1[self.seq1[k1-1]] = self.indx_in_seq1[self.seq1[k1-1]],self.indx_in_seq1[self.seq1[k1]]
        if TEST_COORD_CLAC == 0 :
          self.calc_propped_coords(k)
        else :  
          self.calc_propped_coords_step(k)
        x = self.coords[pk][0]
        if x <= xa:
          break
        k1 = k1-1  

    elif y > ya :
      while True :
        self.seq1[k1+1], self.seq1[k1] = self.seq1[k1], self.seq1[k1+1]
        self.indx_in_seq1[self.seq1[k1]],self.indx_in_seq1[self.seq1[k1+1]] = self.indx_in_seq1[self.seq1[k1+1]],self.indx_in_seq1[self.seq1[k1]]
        if TEST_COORD_CLAC == 0 :
          self.calc_propped_coords(k)
        else :  
          self.calc_propped_coords_step(k)
        y = self.coords[pk][1]
        if y <= ya :
          break
        k1 = k1+1
       

  def calc_indx_in_seq1(self) :
    indx_in_seq1 = np.zeros(self.num_block, dtype=int)
    for i, t in enumerate(self.seq1) :
      indx_in_seq1[t] = i
    return indx_in_seq1  


  def calc_propped_coords(self, k) :
    """
    calculate the coordinates up to the kth module of seq2
    """
    
    for i in range(k+1) :
      self.calc_propped_coords_step(i)
    
     
    
    ''' 
    self.coords =  np.zeros((self.num_block, 2))
    
    for i in range(k+1) :
      pi = self.seq2[i]
      rank_i_seq1 = self.indx_in_seq1[pi]
      
      for j in range(i) :
        pj = self.seq2[j]
        rank_j_seq1 = self.indx_in_seq1[pj]

        if rank_i_seq1 > rank_j_seq1:  # pi is right of pj, update x coord
          self.coords[pi][0] = max(self.coords[pi][0],
                                   self.coords[pj][0] + self.blk_size[pj][0])
        else:  # pi is on top of pj
          self.coords[pi][1] = max(self.coords[pi][1],
                                   self.coords[pj][1] + self.blk_size[pj][1])

      if self.is_fixed[pi] :  # fixed module
        self.coords[pi][0] = max(self.coords[pi][0], self.blk_xy[pi][0])
        self.coords[pi][1] = max(self.coords[pi][1], self.blk_xy[pi][1])
    '''  
      
 

  def calc_propped_coords_step(self, i) :
    """
    single step calculation, update the coords of the ith module in seq2
    """
    pi = self.seq2[i]
    rank_i_seq1 = self.indx_in_seq1[pi]
    self.coords[pi] = [0, 0]

    for j in range(i) :
      pj = self.seq2[j]
      rank_j_seq1 = self.indx_in_seq1[pj]

      if rank_i_seq1 > rank_j_seq1:  # pi is right of pj, update x coord
        self.coords[pi][0] = max(self.coords[pi][0],
                             self.coords[pj][0] + self.blk_size[pj][0])
      else:  # pi is on top of pj
        self.coords[pi][1] = max(self.coords[pi][1],
                                   self.coords[pj][1] + self.blk_size[pj][1])

    if self.is_fixed[pi] :  # fixed module
      self.coords[pi][0] = max(self.coords[pi][0], self.blk_xy[pi][0])
      self.coords[pi][1] = max(self.coords[pi][1], self.blk_xy[pi][1])



  def topo_sort_seq2(self, seq2) :
    return sorted(seq2, key=cmp_to_key(self.dom_comparator))

  
  #def is_free_blk(self, i) :
  #  return self.blks_info[i][1][0] == None

  @staticmethod 
  def is_dominate(x1,y1, x2,y2, w2, h2) :
    """
    determine if module 1 dominate module 2
    """  
    return x1 < x2 + w2 and y1 < y2 + h2


  @staticmethod
  def move_elem_left(seq, j, i) :
    """
    move the jth elem of seq to before the elem in the i the pos 
    """
    x = seq[j]
    for k in range(j, i, -1) :
      seq[k] = seq[k-1]
    seq[i] = x  
 

  def dom_comparator(self, indx1, indx2) :

    # free modules do not have dominance relations
    if (not self.is_fixed[indx1]) or (not self.is_fixed[indx2]) :
      return 0

    if self.is_dominate(self.blk_xy[indx1][0], self.blk_xy[indx1][1], 
              self.blk_xy[indx2][0],  self.blk_xy[indx2][1], 
              self.blk_size[indx2][0], self.blk_size[indx2][1]) :
      return -1

    if self.is_dominate(self.blk_xy[indx2][0], self.blk_xy[indx2][1], 
              self.blk_xy[indx1][0],  self.blk_xy[indx1][1], 
              self.blk_size[indx1][0], self.blk_size[indx1][1]) :
      return 1

    return 0


  def plot(self, show_annotation=True) :
    fig, ax = plt.subplots()
    INF_NEG, INF = -1e100, 1e00
    xmin, ymin, xmax, ymax = INF, INF, INF_NEG, INF_NEG
    for i in range(self.num_block):
      #w = self.blks_info[i][2][0]
      #h = self.blks_info[i][2][1]
      w, h = self.blk_size[i]
      xl = self.coords[i][0]
      yl = self.coords[i][1]
     

      #color = color_map[self.dev_type[i]]
      color = 'blue'
      hatch = None

      if self.is_fixed[i] :
        color = 'red'
        hatch = '.'
      # the physical placeable
      ax.add_artist(Rectangle((xl, yl), w, h, facecolor=color, color=None,
                              edgecolor=None, fill=True))
      # the bounding box
      ax.add_artist(Rectangle((xl, yl), w, h, facecolor=None, color=None,
                              edgecolor='black', fill=False, hatch=hatch))
      # print(xl,yl, w, h, self.dev_type[i])
      cx = xl + w / 2.0
      cy = yl + h / 2.0
      fontsize = 10
      eq_aspect = True
      if show_annotation:
        ax.annotate(self.blks_info[i][0], (cx, cy), color='white',
                    fontsize=fontsize, ha='center', va='center')

      xmin = min(xmin, xl)
      ymin = min(ymin, yl)
      xmax = max(xmax, xl + w)
      ymax = max(ymax, yl + h)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if eq_aspect:
      ax.set_aspect('equal')  




if __name__ == '__main__':
  
  blks_info = [('a', (None, None), (125, 157)),          
               ('b', (None, None), (93,  190)),
               ('c', (None, None), (156, 193) ),
               ('d', (None, None), (129, 95) ),
               ('x', (128, 255), (127, 95)),
               ('y', (190, 0),   (190, 160))]

  netlist_info = []               

  seq1 = [0,2,3,1,5,4]  # acdbyx
  seq2 = [2,0,3,4,5,1]  # cadxyb
  

  
  '''
  blks_info = [
    ('a', (None, None),(1, 2) ),
    ('b', (None, None),(1, 1) ),
    #('c', (None, None),(1, 1) ),
    ('c', (0, 1),(1, 1) ),
    ('d', (None, None),(1, 2) ),
    ('e', (None, None),(1, 2) ),
    ('f', (None, None),(2, 1) ),
    ('g', (None, None),(2, 1) ),
    ('h', (None, None),(3, 1) ),
    ('i', (None, None),(2, 1) ),
    ('j', (None, None),(1, 1) )
  ]



  #seq1 = [8, 9, 7, 0, 2, 1, 3, 4, 6, 5]
  #seq2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

  seq1 = [8, 9, 7, 4, 0, 6, 5, 2, 1, 3]
  seq2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  '''


  spb = seq_pair_blockage(blks_info, netlist_info)
  print(seq1, seq2)
  seq1_res, seq2_res = spb.pack(seq1, seq2, [])
  #print(spb.evaluate())
  seq1  = [spb.blks_info[i][0] for i in seq1]
  seq2  = [spb.blks_info[i][0] for i in seq2]
  seq1_res = [spb.blks_info[i][0] for i in seq1_res]
  seq2_res = [spb.blks_info[i][0] for i in seq2_res]
  print(seq1, seq2)
  print(seq1_res, seq2_res)


  for i in range(spb.num_block) :
    print(spb.blks_info[i][0],  spb.coords[i])

  spb.plot()

  plt.show()

  #seq1_apt, seq2_apt = spb.pack(seq1, seq2)
