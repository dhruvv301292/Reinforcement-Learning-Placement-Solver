#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: seq_operators.py
Created on: Aug 29 3:15 PM 2019
@author: yanglu
"""

# all the routines in this file
# deal with sequence that is a permutation of list(range(N))
# Ref: "Genetic Algorithm for Traveling Salesman Problem with Modified Cycle Crossover Operator"
#       "Analyzing the Performance of Mutation Operators to Solve the Travelling Salesman Problem"
#       "Packing-based VLSImodule placement using genetic algorithm with sequence-pair representation"
#       "Genetic algorithm based two-dimensional and three-dimensional floorplanning for VLSI ASICs"

import random

################## 1 sequence mutation operator  ############################
def two_swap(ind, i, j):
  """
  :param ind:  list or np array
  :param i:
  :param j:
  :return:
  """
  ind[i], ind[j] = ind[j], ind[i]
  return ind


def three_swap(ind, i, j, k):
  """
  :param ind: list or np array
  :param i:
  :param j:
  :param k:
  :return:
  """
  ind[i], ind[j], ind[k] = ind[k], ind[i], ind[j]
  return ind


def seq_rev(ind, i, j):
  """
  :param ind: list or np array
  :param i:
  :param j:
  :return:
  """
  while i < j:
    ind[i], ind[j] = ind[j], ind[i]
    i += 1
    j -= 1
  return ind


def partial_rand_shuffle(ind, indpb):
  ind_size = len(ind)
  for i in range(ind_size):
    if random.random() < indpb:
      j = random.randint(0, ind_size - 1)
      ind[i], ind[j] = ind[j], ind[i]
  return ind

################## 1 sequence cross over operator  ##############################
def order_cx(ind1, ind2, a, b):
  """
  order crossover
  :param ind1: list or np array
  :param ind2: list or np array
  :return:
  """
  ind_size = len(ind1)
  # reverse look up table
  indx1 = [0] * ind_size
  indx2 = [0] * ind_size
  for i, x in enumerate(ind1):
    indx1[x] = i
  for i, x in enumerate(ind2):
    indx2[x] = i


  ind2_cpy = ind2.copy()
  ind1_cpy = ind1.copy()

  for k in range(a, b + 1):
    ind2_cpy[indx2[ind1[k]]] = -1
    ind1_cpy[indx1[ind2[k]]] = -1

  ptr1 = ptr2 = (b + 1) % ind_size

  for i in range(ind_size - (b - a + 1)):

    while ind2_cpy[ptr2] == -1:
      ptr2 = (ptr2 + 1) % ind_size
    while ind1_cpy[ptr1] == -1:
      ptr1 = (ptr1 + 1) % ind_size

    ind1[(b + i + 1) % ind_size] = ind2_cpy[ptr2]
    ind2[(b + i + 1) % ind_size] = ind1_cpy[ptr1]

    ptr1 = (ptr1 + 1) % ind_size
    ptr2 = (ptr2 + 1) % ind_size

  return ind1, ind2


def pmx_cx(ind1, ind2, a, b):
  """
  partially mapped cross over
  :param ind1: list or np array
  :param ind2: list or np array
  :return:
  """

  ind_size = len(ind1)
  # reverse look up table
  indx1 = [0] * ind_size
  indx2 = [0] * ind_size
  for i, x in enumerate(ind1):
    indx1[x] = i
  for i, x in enumerate(ind2):
    indx2[x] = i


  for i in range(a, b+1) :
    ind1[i], ind2[i] = ind2[i], ind1[i]

  # the remaining position to fill
  irange = list(range(0,a)) + list(range(b+1, ind_size))

  for i in irange :
    x = ind1[i]
    while indx2[x] >=a and indx2[x] <= b :
      x = ind2[indx2[x]]
    ind1[i] = x

    x = ind2[i]
    while indx1[x] >= a and indx1[x] <= b:
      x = ind1[indx1[x]]
    ind2[i] = x

  return ind1, ind2


def cycle_cx(ind1, ind2):
  """
  cycle crossover
  :param ind1: list or np array
  :param ind2: list or np array
  :return:
  """
  ind_size = len(ind1)
  o1 = [-1]*ind_size
  o2 = [-1]*ind_size

  # reverse look up table
  indx1 = [0] * ind_size
  indx2 = [0] * ind_size
  for i, x in enumerate(ind1):
    indx1[x] = i
  for i, x in enumerate(ind2):
    indx2[x] = i

  i = 0
  o1[i] = ind1[i]
  i = indx1[ind2[i]]
  while o1[i] == -1 :
    o1[i] = ind1[i]
    i = indx1[ind2[i]]
  for i in range(ind_size) :
    if o1[i] == -1:
      o1[i] = ind2[i]

  i = 0
  o2[i] = ind2[i]
  i = indx2[ind1[i]]
  while o2[i] == -1:
    o2[i] = ind2[i]
    i = indx2[ind1[i]]
  for i in range(ind_size):
    if o2[i] == -1:
      o2[i] = ind1[i]

  # we want in place change of ind1 and ind2
  ind1[:] = o1[:]
  ind2[:] = o2[:]


  return ind1, ind2

############## sequence-pair mutation operator ###########################
def seq_pair_swap_batch(pos_seq, neg_seq, a_indx, b_indx) :
  """
  physically swap blocks given by a_indx to blocks given by b_indx
  both a_indx and b_indx corresponds to pos_pair
  :param pos_pair:
  :param neg_seq:
  :return:
  """
  N = len(pos_seq)
  rev_indx_tab = [0] *N
  for i, v in enumerate(neg_seq):
    rev_indx_tab[v] = i

  for n in range(len(a_indx)) :
    i = a_indx[n]
    j = b_indx[n]
    ip = rev_indx_tab[pos_seq[i]]
    jp = rev_indx_tab[pos_seq[j]]
    pos_seq[i],pos_seq[j] = pos_seq[j],pos_seq[i]
    neg_seq[ip], neg_seq[jp] = neg_seq[jp], neg_seq[ip]


  return pos_seq, neg_seq


def seq_pair_swap(pos_seq, neg_seq, i, j) :
  xi = pos_seq[i]
  xj = pos_seq[j]

  ip, jp = 0, 0
  for n in range(len(neg_seq)) :
    if neg_seq[n] == xi:
      ip = n
    if neg_seq[n] == xj :
      jp = n

  pos_seq[i], pos_seq[j] = pos_seq[j], pos_seq[i]
  neg_seq[ip], neg_seq[jp] = neg_seq[jp], neg_seq[ip]



def seq_pair_rotate(pos_seq, neg_seq, i, j, mode) :

  """
  swap block only in pos pair or in neg pair, controlled by mode
  :param pos_seq:
  :param neg_seq:
  :param i:
  :param j:
  :param mode:
  :return:
  """
  if mode == 0 :
    pos_seq[i], pos_seq[j] = pos_seq[j], pos_seq[i]
  else :
    neg_seq[i], neg_seq[j] = neg_seq[j], neg_seq[i]


def seq_pair_reverse(pos_seq, neg_seq, indx) :
  """
  physically reverse the order of sub blocks given by index
  index corresponds to pos_seq
  :param pos_seq:
  :param neg_seq:
  :param index:
  :return:
  """

  N = len(pos_seq)
  rev_indx_tab = [0] * N
  for i, v in enumerate(neg_seq):
    rev_indx_tab[v] = i

  s, e = 0, len(indx)-1
  while s < e :
    i = indx[s]
    j = indx[e]
    ip = rev_indx_tab[pos_seq[i]]
    jp = rev_indx_tab[pos_seq[i]]
    pos_seq[i], pos_seq[j] = pos_seq[j], pos_seq[i]
    neg_seq[ip], neg_seq[jp] = neg_seq[jp], neg_seq[ip]
    s += 1
    e -= 1

  return pos_seq, neg_seq



############## sequence-pair cross over operator ###########################
def seq_pair_order_cx_helper(p, indx_p, p_msk, rev_tab) :
  """
  helper func for seq_pair_order_cx
  :param p:  sequence to work with
  :param indx_p: index of p to be preserved
  :param p_msk: sequence of another individual
  :param rev_tab: reverse table to look up index given value, for p_msk
  :return: p modified in place
  """
  N = len(p)
  p_cpy = p.copy()
  p[:] = [-1] * N
  for i in indx_p:
    p[i] = p_cpy[i]
    p_msk[rev_tab[p[i]]] = -1
  # combine p and p_msk to form new p
  j = 0
  for i in range(N):
    if p[i] != -1:
      continue
    while p_msk[j] == -1:
      j += 1
    p[i] = p_msk[j]
    j += 1


def seq_pair_order_cx(p1, s1, p2, s2, a, b) :
  """
  a topology preserving cross over for sequence pair
  :param p1: primary sequence for SP1
  :param s1: secondary sequence for SP1
  :param p2: primary sequence for SP2
  :param s2: secondary sequence for SP2
  :param a:
  :param b:
  :return:
  """

  N = len(p1)
  rev_p1 = [0] * N
  rev_p2 = [0] * N
  rev_s1 = [0] * N
  rev_s2 = [0] * N

  for i in range(N) :
    rev_p1[p1[i]] = i
    rev_p2[p2[i]] = i
    rev_s1[s1[i]] = i
    rev_s2[s2[i]] = i

  # make copy
  p1_msk = p2.copy()
  p2_msk = p1.copy()
  s1_msk = s2.copy()
  s2_msk = s1.copy()

  indx_p1 = list(range(a, b + 1))
  indx_s1 = [rev_s1[p1[i]] for i in indx_p1]
  indx_p2 = list(range(a, b + 1))
  indx_s2 = [rev_s2[p2[i]] for i in indx_p2]


  # working on p1
  seq_pair_order_cx_helper(p1,indx_p1,p1_msk,rev_p2)

  # working on p2
  seq_pair_order_cx_helper(p2,indx_p2, p2_msk, rev_p1)

  # working on s1
  seq_pair_order_cx_helper(s1,indx_s1, s1_msk, rev_s2)

  # working on s2
  seq_pair_order_cx_helper(s2,indx_s2,s2_msk, rev_s1)

  return p1, s1, p2, s2, p1_msk, p2_msk



if __name__ == '__main__' :

  print('------testing sequence order cross over------')
  s1 = [2, 3, 7, 1, 6, 0, 5, 4]
  s2 = [3, 1, 4, 0, 5, 7, 2, 6]
  print('s1, s2 before cross over')
  print(s1, s2)
  s1,s2 = order_cx(s1,s2, 3,5)
  print('s1, s2 after cross over')
  print(s1, s2)

  print('------testing sequence partially mapped cross over------')
  s1 = [2, 3, 7, 1, 6, 0, 5, 4]
  s2 = [3, 1, 4, 0, 5, 7, 2, 6]
  print('s1, s2 before cross over')
  print(s1, s2)
  s1, s2 = pmx_cx(s1, s2, 3, 5)
  print('s1, s2 after cross over')
  print(s1, s2)

  print('------testing sequence cycle cross over------')
  s1 = [0, 1, 2, 3, 4, 5, 6, 7]
  s2 = [7, 4, 1, 0, 2, 5, 3, 6]
  print('s1, s2 before cross over')
  print(s1, s2)
  s1, s2 = cycle_cx(s1, s2)
  print('s1, s2 after cross over')
  print(s1, s2)



  print('------testing sequence pair order cross over------')
  p1 = [0,1,2,3,4,5,6,7]
  s1 = [7,6,5,4,3,2,1,0]

  p2 = [1,3,6,0,2,7,5,4]
  s2 = [0,7,1,6,2,5,3,4]
  print('p1, s1, p2, s2 before cx')
  print(p1, s1)
  print(p2, s2)
  print('p1, s1, p2, s2 after cx')
  p1, s2, p2, s2, _, _ = seq_pair_order_cx(p1,s1,p2,s2, 1,3)
  print(p1, s1)
  print(p2, s2)


