#include <iostream>
#include <utility>
#include <vector>
#include <algorithm>
#include <ctime>
#include "fastseqpair.hpp"




using namespace std;


void gen_lattice_placement_input(int n, double blk_width,
     std::vector<std::vector<double> > & blk_size,
      std::vector<std::vector<int> > & net_blk_pair)
{
  int net_count = 0;
  blk_size.clear();
  blk_size.resize(n*n, vector<double>(2, blk_width));
  net_blk_pair.clear();
  for (int r=0; r<n; ++r) 
    for (int c=0; c<n; ++c) {
      int indx = r*n + c;
      if (r < n-1)
        net_blk_pair.push_back({indx, indx + n});

      if (c < n-1) 
        net_blk_pair.push_back({indx, indx+1});
    } 

}



int main()
{


  int n = 10;
  double blk_width = 4;

  vector<vector<double> > blk_size;
  vector<vector<int> > net_blk_pair;

  gen_lattice_placement_input(n, blk_width, blk_size, net_blk_pair);

  FastSeqPair sp(blk_size, net_blk_pair);

  double w_area = 0.0;
  double w_wireLen = 1.0;
  sp.setAreaWireLenWeights(w_area, w_wireLen);

  int N = n * n;
  vector<int> seq1(N, 0);
  for (int i=0; i<N; ++i) seq1[i] = i;

  vector<int> seq2 = seq1; 

  random_shuffle(seq1.begin(), seq1.end());
  random_shuffle(seq2.begin(), seq2.end());
  
  //seq1 = {4, 28, 11, 15, 24, 35, 17, 29, 26, 31, 33, 22, 19, 2, 0, 23, 25, 18, 32, 8, 5, 21, 7, 12, 14, 20, 27, 6, 10, 1, 30, 9, 13, 34, 3, 16};
  //seq2 = {19, 11, 30, 34, 5, 33, 26, 2, 20, 31, 32, 6, 7, 27, 12, 13, 22, 21, 28, 10, 18, 24, 15, 25, 8, 16, 4, 29, 3, 35, 23, 9, 17, 14, 0, 1}; 


  clock_t begin = clock();
  for (int i=0; i<10000; ++i) {
    sp.pack(seq1, seq2);
    sp.evaluate();
  }
  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  cout << "elapsed sec : " << elapsed_secs << endl;
  return 0;

  std::vector<std::vector<double> >  coords = sp.get_coords();

  for (int i=0; i<N; ++i)
    cout <<seq1[i] <<", ";
  cout << endl;

  for (int i=0; i<N; ++i)
    cout <<seq2[i] <<", ";  
  cout << endl;

  for (int i=0; i<N; ++i)
    cout << coords[i][0]  << ' ' << coords[i][1] << endl;



  


}


