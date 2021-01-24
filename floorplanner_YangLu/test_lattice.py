import csv
from seq_pair_obs import seq_pair_blockage
from seq_pair_fast import seq_pair_fast
from seq_pair_fast_cpp import seq_pair_fast_cpp
from seq_pair_optimizer import SeqPairOpt
from seq_pair_sa import SeqPairSA
import random
import numpy as np
import matplotlib.pyplot as plt
from lattice_place_problem import *
import pprint
import time
import sys
import os
sys.path.append('/home/yanglu/move37/prototype/Floorplanner/swig')
from fastseqpair import FastSeqPair as cFastSeqPair




# TODO: search parallel simulated annealing alg
seed = 25 #25
random.seed(seed)
np.random.seed(seed)


params_go = {
 'population' : 300,
 'max_gen'    : 200,   #200,
 'xcpb'       : 0.1,
 'mtpb'       : 0.8,
 'indpb_seq'      : 0.05, 
 'indpb_shape'    : 0.1,
 'shape_gene_enable' : True,
 'tournsize'         : 10
}


n = 10 #4
size = 4
blk_info,  netlist_info = gen_lattice_placement_input(n, size)
blk_info[0] = ('d-0-0', (0,0), (4,4))


pp = pprint.PrettyPrinter(indent=2)
pp.pprint(blk_info)
#pp.pprint(netlist_info)
for info in netlist_info :
  indx1 = info[1][0]
  indx2 = info[1][1]
  print(blk_info[indx1][0],blk_info[indx2][0])

# original SP algorithm
#sp = seq_pair_blockage(blk_info, netlist_info, do_adapt = True)
#sp = seq_pair_blockage(blk_info, netlist_info, do_adapt = False)
#sp.set_area_wireLen_weights(0.0, 1.0)


#sp = seq_pair_fast(blk_info, netlist_info)  # fast SP algorithm with python
sp = seq_pair_fast_cpp(blk_info, netlist_info) # fast SP alg with C++ and python binding
sp.set_area_wireLen_weights(0.0, 1.0)


# test, use swig wrapped c++ code
#blk_size = [ (t[2][0], t[2][1]) for t in blk_info]
#net_blk_pair = [ (t[1][0], t[1][1]) for t in netlist_info]
#sp = cFastSeqPair(blk_size, net_blk_pair)
#sp.setAreaWireLenWeights(0.0, 1.0)


'''
go = SeqPairOpt(sp, **params_go, save_num=1)
go.set_policy()
_,_,hof = go.evolve()
go.report(hof)
'''



num_block = len(blk_info)
seq1 = random.sample(range(num_block), num_block)
seq2 = random.sample(range(num_block), num_block)
shape = [random.randint(0,1) for _ in range(num_block)]
init_state = seq1 + seq2 + shape 


# test
'''
start = time.time()
for i in range(1000) :
  sp.pack(seq1,seq2,shape)
  sp.evaluate()
end = time.time()
print('cost = ', sp.evaluate())
print('elapsed time: ', end - start)
exit()
'''


sa = SeqPairSA(init_state, sp)

#minutes = 3 # 3
#sa.set_schedule(sa.auto(minutes=minutes))


sa.Tmax = 400  # Max (starting) temperature
sa.Tmin = 0.25   # Min (ending) temperature
sa.steps = 1000000   # Number of iterations
sa.updates = 100   # Number of updates (by default an update prints to stdout)


sa.copy_strategy = "slice"
state, e = sa.anneal()

print('final energy = ', e)
print('state = ', state)

seq1_res = state[:sp.num_block]
seq2_res = state[sp.num_block:2*sp.num_block]
shape_res = state[2*sp.num_block:]

sp.pack(seq1_res,seq2_res, shape_res)
sp.plot()


plt.show()

