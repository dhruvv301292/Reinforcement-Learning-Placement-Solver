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


class seq_pair_fast :
  def __init__(self, blocks_info, netlist_info) :
    self.blks_info = blocks_info
    self.netlist_info = netlist_info
    self.num_block = len(self.blks_info)
    self.coords =  np.zeros((self.num_block, 2))

    #self.do_adapt = do_adapt
    
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


    self.net_blk_pair = np.zeros((len(self.netlist_info), 2), dtype=int)
    for i in range(len(self.netlist_info)) :
      self.net_blk_pair[i] = self.netlist_info[i][1]  

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
    return self.xmax * self.ymax


  def calc_wireLen(self) :
    '''
    cost = 0
    for info in self.netlist_info :
      indx1 = info[1][0]
      indx2 = info[1][1]
      x1,y1 = self.coords[indx1]
      w1,h1 = self.blk_size[indx1]
      x2,y2 = self.coords[indx2]
      w2,h2 = self.blk_size[indx2]

      cost += abs((x2+w2/2) -  (x1+w1/2)) + abs((y2+h2/2) -  (y1+h1/2)) 


    return cost'''

    coords = self.coords + self.blk_size/2
    dxy = coords[self.net_blk_pair[:,0], :] -  coords[self.net_blk_pair[:,1], :]
    return sum(sum(np.abs(dxy)))    



  def pack(self, seq1, seq2, shape=[]) :
    self.coords =  np.zeros((self.num_block, 2))
    self.set_shape(shape)  

    seq1_rev = [seq1[self.num_block-1-i] for i in range(self.num_block)]
    #match1 = np.zeros(self.num_block,dtype=int)
    #match1_rev = np.zeros(self.num_block,dtype=int)
    match2 = np.zeros(self.num_block,dtype=int)

    for i in range(self.num_block) :
      #match1[seq1[i]] = i
      match2[seq2[i]] = i
      #match1_rev[seq1_rev[i]] = i

    #print(seq1, seq1_rev, seq2)
    Lx = np.zeros(self.num_block) 

    for i in range(self.num_block) :
      b = seq1[i]
      p = match2[b]
      self.coords[b][0] = Lx[p]  # x cooord
      t = Lx[p] + self.blk_size[b][0]

      for j in range(p, self.num_block) :
        if t > Lx[j] :
          Lx[j] = t
        else :
          break 

    Ly = np.zeros(self.num_block) 

    for i in range(self.num_block) :
      b = seq1_rev[i]
      p = match2[b]
      self.coords[b][1] = Ly[p]  # x cooord
      t = Ly[p] + self.blk_size[b][1]

      for j in range(p, self.num_block) :
        if t > Ly[j] :
          Ly[j] = t
        else :
          break      


    self.xmax = Lx[self.num_block-1]
    self.ymax = Ly[self.num_block-1]


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



  blks_info = [
    ('a', (None, None),(1, 2) ),
    ('b', (None, None),(1, 1) ),
    ('c', (None, None),(1, 1) ),
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

  netlist_info = []
  

  spb = seq_pair_fast(blks_info, netlist_info)
  #print(seq1, seq2)
  spb.pack(seq1, seq2, [])
  spb.plot()
  plt.show()

