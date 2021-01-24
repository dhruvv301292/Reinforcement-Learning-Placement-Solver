#include "fastseqpair.hpp"


using namespace std;





FastSeqPair::FastSeqPair(std::vector<std::vector<double> > const & blk_size,
      std::vector<std::vector<int> > const & net_blk_pair)

{
  _blk_size_orig = blk_size;
  _blk_size = blk_size;
  _net_blk_pair = net_blk_pair;
  _nBlks = _blk_size.size();
  _coords.clear();
  _coords.resize(_nBlks);


  _w_area = 0.5;
  _w_wireLen = 0.5;

}


void FastSeqPair::set_shape(vector<int> const & shape) {

  _blk_size = _blk_size_orig;
  for (int i=0; i<_nBlks; ++i) 
    if (shape[i] == 1) { // flip
      _blk_size[i][0] = _blk_size_orig[i][1];
      _blk_size[i][1] = _blk_size_orig[i][0];

    } 
}

double FastSeqPair::calc_wireLen()
{

  double cost = 0;
  for (int i=0; i<_net_blk_pair.size(); ++i) {
    int indx1 = _net_blk_pair[i][0];
    int indx2 = _net_blk_pair[i][1];
    double x1c = _coords[indx1][0] + _blk_size[indx1][0]/2;
    double y1c = _coords[indx1][1] + _blk_size[indx1][1]/2;
    double x2c = _coords[indx2][0] + _blk_size[indx2][0]/2;
    double y2c = _coords[indx2][1] + _blk_size[indx2][1]/2;
    cost += fabs(x1c - x2c) + fabs(y1c - y2c);
  }
  return cost;
}



void FastSeqPair::pack(std::vector<int> const & seq1, std::vector<int> const & seq2)
{
  //_coords.clear();
  //_coords.resize(_nBlks, pair<double,double>(0,0, 0.0));

  // clean coords
  fill(_coords.begin(), _coords.end(), vector<double>(2,0));

  vector<int> seq1_rev(_nBlks, 0);
  vector<int> match2(_nBlks, 0);

  for (int i=0; i<_nBlks; ++i) {
    seq1_rev[i] = seq1[_nBlks-1-i];
    match2[seq2[i]] = i;
  }

  vector<double> L(_nBlks, 0);

  // calc x coord 
  for (int i=0; i<_nBlks; ++i) {
    int b = seq1[i];
    int p = match2[b];
    _coords[b][0] = L[p]; // x coord
    double t = L[p] + _blk_size[b][0];

    for (int j=p; j<_nBlks; ++j) {
      if (t > L[j])
        L[j] = t;
      else 
        break;
    }
  }
  _xmax = L[_nBlks-1];

  // calc y coord 
  fill(L.begin(), L.end(), 0);
  for (int i=0; i<_nBlks; ++i) {
    int b = seq1_rev[i];
    int p = match2[b];
    _coords[b][1] = L[p]; // x coord
    double t = L[p] + _blk_size[b][1];

    for (int j=p; j<_nBlks; ++j) {
      if (t > L[j])
        L[j] = t;
      else 
        break;
    }
  }

  _ymax = L[_nBlks-1];

}


