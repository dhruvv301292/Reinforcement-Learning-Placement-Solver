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

import sys
sys.path.append('/home/yanglu/move37/prototype/Floorplanner/swig')
# sys.path.append('/Users/liaohaiguang/Desktop/DARPA\\ IDEA/1.Project/Github/RL_Placement/main/floorplanner_YangLu/swig')

from fastseqpair import FastSeqPair as cFastSeqPair

# class that wrap c++ implementation of fast sequence pair
class seq_pair_fast_cpp :
  def __init__(self, blocks_info, netlist_info) :
    self.blks_info = blocks_info
    self.netlist_info = netlist_info
    self.num_block = len(self.blks_info)
    
    #self.coords =  np.zeros((self.num_block, 2))
    self.coords = []

    #self.do_adapt = do_adapt
    self.is_fixed = np.zeros(self.num_block, dtype=bool) 

    # initial position
    self.blk_xy   = np.zeros((self.num_block, 2))

    self.w_area = 0.5
    self.w_wireLen = 0.5
    
    for i in range(self.num_block) :
      if self.blks_info[i][1][0] != None :
        self.is_fixed[i] = True

    blk_size = [ (t[2][0], t[2][1]) for t in self.blks_info]
    self.net_blk_pair = [ (t[1][0], t[1][1]) for t in netlist_info]
    self.sp_cpp = cFastSeqPair(blk_size, self.net_blk_pair)  


  def set_area_wireLen_weights(self, w_area, w_wireLen) :
    self.sp_cpp.setAreaWireLenWeights(w_area, w_wireLen)

  

  def evaluate(self) :
    """
    evaluate the quality of placement, by default
    return the area of the minmum bounding box
    """
    return self.sp_cpp.evaluate()
    

  def pack(self, seq1, seq2, shape=[]) :
    if len(shape) != 0 :
        self.sp_cpp.set_shape(shape)
    self.sp_cpp.pack(seq1,seq2)


  def plot(self, show_annotation=True) :
    fig, ax = plt.subplots()
    INF_NEG, INF = -1e100, 1e00
    xmin, ymin, xmax, ymax = INF, INF, INF_NEG, INF_NEG
    coords = self.sp_cpp.get_coords()
    blk_size = self.sp_cpp.get_blk_size()

    for i in range(self.num_block):
      #w = self.blks_info[i][2][0]
      #h = self.blks_info[i][2][1]
      w, h = blk_size[i]
      xl = coords[i][0]
      yl = coords[i][1]
     

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
  

  spb = seq_pair_fast_cpp(blks_info, netlist_info)
  #print(seq1, seq2)
  spb.pack(seq1, seq2, [])
  spb.plot()
  plt.show()

