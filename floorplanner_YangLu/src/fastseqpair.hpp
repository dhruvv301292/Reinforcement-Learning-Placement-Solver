#include <vector>
#include <algorithm>
#include <utility>



class FastSeqPair
{
  public:
    FastSeqPair(std::vector<std::vector<double > > const & blk_size,
      std::vector< std::vector<int> > const & net_blk_pair);


    void set_shape(std::vector<int> const & shape);

    void setAreaWireLenWeights(double w_area, double w_wireLen) {
      _w_area = w_area;
      _w_wireLen = w_wireLen;
    }
    
    std::vector<std::vector<double> > get_coords() {
      return _coords;
    }

    std::vector<std::vector<double> > get_blk_size() {
      return _blk_size;
    }

    double evaluate() {
      return _w_area * calc_area() + _w_wireLen * calc_wireLen();
    }

    double calc_area() {
      return _xmax * _ymax;
    }

    double calc_wireLen();

    void pack(std::vector<int> const & seq1, std::vector<int> const & seq2);
  private:
    double _xmax;
    double _ymax;
    double _w_area;
    double _w_wireLen;
    int _nBlks;
    std::vector<std::vector<double> > _blk_size_orig; // original blk size
    std::vector<std::vector<double> > _blk_size; // blk size after reshape
    std::vector<std::vector<int> > _net_blk_pair;
    //std::vector<std::pair<double,double> > _coords;
    std::vector<std::vector<double> > _coords;

};