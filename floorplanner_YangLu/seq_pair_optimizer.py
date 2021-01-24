#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Filename: seq_pair_optimizer.py
Created on: Seq 2 5:33 PM 2019

@author: yanglu
"""

import random

import matplotlib.pyplot as plt
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from seq_operators import *
from seq_pair_obs import seq_pair_blockage

class SeqPairOpt :
  def __init__(self, sp: seq_pair_blockage, population, xcpb, mtpb,
               max_gen, indpb_seq=0.2, indpb_shape=0.2, tournsize=10,
               shape_gene_enable=True, save_num=1):
    self.sp = sp
    self.num_block = sp.num_block
    self.toolbox = base.Toolbox()
    self.num_population = population
    self.xcpb = xcpb
    self.mtpb = mtpb
    self.num_gen = max_gen
    self.indpb_seq = indpb_seq
    self.indpb_shape = indpb_shape
    self.tournsize = tournsize
    self.shape_gene_enable = shape_gene_enable
    self.save_num = save_num

  def set_policy(self):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    self.toolbox.register("population", tools.initRepeat, list, self.gen_rand_indvidual)
    self.toolbox.register("mate", self._seq_pair_cx)
    self.toolbox.register("mutate", self._seq_pair_mutate, indpb_seq=self.indpb_seq,
                          indpb_shape=self.indpb_shape)
    self.toolbox.register("select", tools.selTournament, tournsize=self.tournsize)
    self.toolbox.register("evaluate", self._evaluate)


  ## TODO: add the best init guess as argument
  # def evolve(self, best_init_individual) ...
  def evolve(self):

    pop = self.toolbox.population(n=self.num_population)
    hof = tools.HallOfFame(self.save_num, similar=np.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, self.toolbox,
                        self.xcpb, self.mtpb, self.num_gen, stats=stats, halloffame=hof)
    return pop, stats, hof

  def report(self, hof):
    for i in range(len(hof)):
      '''
      self.sp.set_pos_seq(hof[i][:self.num_block])
      self.sp.set_neg_seq(hof[i][self.num_block:2 * self.num_block])

      if self.shape_gene_enable:
        self.sp.set_shape_seq(hof[i][2 * self.num_block:])
      self.sp.pack()
      '''
      seq1 = hof[i][:self.num_block]
      seq2 = hof[i][self.num_block:2*self.num_block]
      shape = []
      if self.shape_gene_enable:
        shape = hof[i][2*self.num_block:]

      
      self.sp.pack(seq1, seq2, shape)
      print(' i = ', i)
      print('gene :', hof[i])
      print('coords:', self.sp.coords)
      print('cost:', self.sp.evaluate())
      self.sp.plot()
    #plt.show()


  def _evaluate(self, ind):
    seq1 = ind[:self.num_block]
    seq2 = ind[self.num_block:2 * self.num_block]
    shape = []
    if self.shape_gene_enable :
      shape = ind[2*self.num_block:]
    self.sp.pack(seq1, seq2, shape)
    return self.sp.evaluate(),


  def gen_rand_indvidual(self):
    pos_seq = random.sample(range(self.num_block), self.num_block)
    neg_seq = random.sample(range(self.num_block), self.num_block)
    shape_seq = [random.randint(0,1) for _ in range(self.num_block)]  # no shape gene
    return creator.Individual(pos_seq + neg_seq + shape_seq)

  def _seq_pair_cx(self, ind1, ind2) :

    a, b = random.sample(range(self.num_block), 2)
    if a > b:
      a, b = b, a

    p1 = ind1[0:self.num_block]
    s1 = ind1[self.num_block:2*self.num_block]
    p2 = ind2[0:self.num_block]
    s2 = ind2[self.num_block:2*self.num_block]

    r_num = random.random()
    # randomly switch primary seq and secondary seq
    if r_num < 0.5 :
      p1,s1 = s1, p1
      p2,s2 = s2, p2

    shape1 = ind1[2*self.num_block:]  
    shape2 = ind2[2*self.num_block:]  

    p1, s1, p2, s2, p1_msk, p2_msk = seq_pair_order_cx(p1,s1,p2,s2,a,b)

    if r_num >= 0.5 :
      ind1[0:self.num_block] = p1
      ind1[self.num_block:2 * self.num_block] = s1
      ind2[0:self.num_block] = p2
      ind2[self.num_block:2 * self.num_block] = s2


    else :
      ind1[0:self.num_block] = s1
      ind1[self.num_block:2 * self.num_block] = p1
      ind2[0:self.num_block] = s2
      ind2[self.num_block:2 * self.num_block] = p2

    # the cross over of shape gene need to be consistent with order gene

    for i in p1_msk:
      if i != -1:
        ind1[2 * self.num_block + i] = shape2[i]
    for i in p2_msk:
      if i != -1:
        ind2[2 * self.num_block + i] = shape1[i]
    

    return ind1, ind2


  def _seq_pair_mutate(self, ind, indpb_seq, indpb_shape):

    # mutate sequence gene
    N = int(indpb_seq * self.num_block)
    pos_seq = ind[0:self.num_block]
    neg_seq = ind[self.num_block:2*self.num_block]
    shape_gege = ind[2*self.num_block:]

    pool = list(range(self.num_block))
    indx_a = random.sample(pool, N)
    indx_b = random.sample(pool, N)

    for n in range(N) :
      i = indx_a[n]
      j = indx_b[n]

      dice = random.random()
      if dice < 0.33 : # swap
        seq_pair_swap(pos_seq,neg_seq, i, j)
      elif dice < 0.66 :  # rotate mode1
        seq_pair_rotate(pos_seq, neg_seq, i,j, 0)
      else :    # rotate mode2
        seq_pair_rotate(pos_seq, neg_seq, i, j, 1)


    ind[0:self.num_block] = pos_seq
    ind[self.num_block:2*self.num_block] = neg_seq

    # mutate shape gene
    if self.shape_gene_enable :
      for i in range(self.num_block) :
        # flip
        if random.random() < indpb_shape :
          ind[2 * self.num_block+ i] = 1 -  ind[2 * self.num_block+ i]
 
    return ind,



    
