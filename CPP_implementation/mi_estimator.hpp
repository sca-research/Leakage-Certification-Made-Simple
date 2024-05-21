#ifndef MI_EST_H_
#define MI_EST_H_

#include <math.h>
#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>

#include "mydist.hpp"


using namespace mlpack;
using namespace mlpack::neighbor; // NeighborSearch and NearestNeighborSort
using namespace mlpack::metric;   // EuclideanDistance

struct mi_t {
  double mi_var;
  double mi_avg;
};

class mi_estimator {

private:

  std::string fname;
  
public:

  double s;
  arma::mat data;
  mi_estimator(const std::string fname, bool no_sigma):
    fname(fname) {

    if( data.load(fname) )
      {
	int nr = data.n_rows;	
	if(!no_sigma)
	  {
	    arma::rowvec vec_key_sigma = data.row(nr-1);
	    data = data.rows(1,nr-1); /* Ignore the last row. It stores parameter values */
	    if(data.n_cols > 1) s = vec_key_sigma[data.n_cols-1];
	  }
	
      }
    else
      printf("Failed to load data from %s\n", fname.c_str()); 
  }

  
  double sigma_from_file() {
    return s;
  }

  friend arma::uvec sample_wor(int, int);
  
  struct mi_t mi_mixed(const int n_sample, int k, const int n_trial);

  
  struct mi_t mi_ksg(const int n_sample, const int k, const int n_trial);
  struct mi_t mi_truncksg(const int n_sample, const int k, const int n_trial);

  struct mi_t mi_biksg(const int n_sample, const int k, const int n_trial);

  arma::vec mi_multiple_sample(const int k);


  ~mi_estimator(){}

};


#endif
