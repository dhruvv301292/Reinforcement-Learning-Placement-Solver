## TODO: need to see simulated anealing 
# https://github.com/perrygeo/simanneal

## TODO: then need to test two adapt strategies as seen in the paper

import csv
from seq_pair_obs import seq_pair_blockage
from seq_pair_optimizer import SeqPairOpt
import random
import numpy as np
import matplotlib.pyplot as plt


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
#blks_info_ami49[0] = ('M001', (2618, 4088), (3234, 1708))
# let's fix module M004
#blks_info_ami49[3] = ('M004', (0, 2044), (1610, 3080))
# let's fix module M005
#blks_info_ami49[4] = ('M005', (2548, 1904), (1386, 532))


# let's fix module M001
blks_info_ami49[0] = ('M001', (0, 630), (3234, 1708))
# let's fix module M004
blks_info_ami49[3] = ('M004', (630, 5040), (3080, 1610))
# let's fix module M006
blks_info_ami49[5] = ('M006', (4774, 2548), (882, 1862))



'''
params_go = {
 'population' : 300, #300,
 'max_gen'    : 200,   #200,
 'xcpb'       : 0.1,
 'mtpb'       : 0.8,
 'indpb_seq'      : 0.05, 
 'indpb_shape'    : 0.05,
 'shape_gene_enable' : True,
 'tournsize'         : 10
}
'''


params_go = {
 'population' : 300, #300,
 'max_gen'    : 200,   #200,
 'xcpb'       : 0.1,
 'mtpb'       : 0.8,
 'indpb_seq'      : 0.05, 
 'indpb_shape'    : 0.05,
 'shape_gene_enable' : True,
 'tournsize'         : 10
}


netlist_info = []
sp = seq_pair_blockage(blks_info_ami49, netlist_info, do_adapt = True)


go = SeqPairOpt(sp, **params_go, save_num=1)
go.set_policy()
_,_,hof = go.evolve()
go.report(hof)


plt.show()

