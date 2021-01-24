# lattice placement example


def gen_lattice_placement_input(n, size) :
  """
  n :  n x n lattice placement example, 
  size: size of each placement block
  return: blk info, net list info
  """

  blk_info = [None for i in range(n*n)]
  netlist_info = []
  net_count = 0
  for r in range(n) :  # row
    for c in range(n) : # col
      name = 'd' + '-' + str(r) + '-' + str(c)
      indx = r * n + c
      blk_info[indx] = (name, (None, None), (size, size))

      if r < n-1:
        indx_top = indx + n
        netlist_info.append( (str(net_count), (indx, indx_top)) )
        net_count += 1  
      
      if c < n-1:
        indx_right = indx + 1
        netlist_info.append( (str(net_count), (indx, indx_right)) )
        net_count += 1    


  return blk_info,  netlist_info

