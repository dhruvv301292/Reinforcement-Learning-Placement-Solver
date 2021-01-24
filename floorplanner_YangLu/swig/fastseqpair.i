%module(direction="1") fastseqpair

%{
#include "fastseqpair.hpp"
%}



%include "std_vector.i"
%include "std_string.i"
%include "std_map.i"
%include "std_pair.i"

/*in swig, vector is converted to list, vector of vector is converted to tuple of tuple*/

namespace std {
 %template(vector_i) vector<int>;
 %template(vector_d) vector<double>;
 %template(vector_s) vector<string>;
 %template(vecvec_i) vector<vector<int> >;
 %template(vecvec_d) vector<vector<double> >;
 %template(vecpair_i) vector<pair<int,int> >;
 %template(vecpair_d) vector<pair<double, double> >;
}

%include "fastseqpair.hpp"
