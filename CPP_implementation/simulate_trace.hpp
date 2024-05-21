#ifndef SIMUL_TRC_
#define SIMUL_TRC_

#include <random>
#include <boost/random/laplace_distribution.hpp>

#include "sbox.hpp" /* AES Sbox definition */

#ifndef MODEL_F_
#define MODEL_F_
unsigned int hamming_wt(const unsigned int a); 
unsigned int identy(const unsigned int a);

unsigned int nonlinear_sb(const unsigned int a); 
unsigned int square(const unsigned int a);
unsigned int whamming_wt(const unsigned int a);
unsigned int hamming_dist(const unsigned int a, const unsigned int b);

#endif

#ifndef CRYPTO_F_
#define CRYPTO_F_

unsigned int sbox(unsigned int x, unsigned int key); 

#endif


class trace {

private:
  std::string fname;
  std::string model_name;
  unsigned int (*crypto_func)(const unsigned int, const unsigned int);
public:

  explicit trace(const std::string fname, const std::string model_name, unsigned int (*cryptofunc)(const unsigned int, const unsigned int)):
    fname(fname), model_name(model_name)
  {
    crypto_func = cryptofunc;
  }


  //To generate trace using leakage function that takes one input e.g. Hamming weight 

  bool trace1d(const uint32_t n_trc, const unsigned int secret_key,  
		   std::string distr_name, const double sigma, unsigned int (*lkg_func)(const unsigned int));

  //To generate trace using leakage function that takes two inputs e.g. Hamming distance

  bool trace1hd(const uint32_t n_trc, const unsigned int secret_key,
	       std::string distr_name, const double sigma, unsigned int (*lkg_func)(const unsigned int, const unsigned int)); 

  bool trace2d_mask(const uint32_t n_trc, const unsigned int secret_key,
		    std::string distr_name, const double sigma, unsigned int (*lkg_func)(const unsigned int));
  
  bool trace1d_mask(const uint32_t n_trc, const unsigned int secret_key,
		    std::string distr_name, const double sigma, unsigned int (*lkg_func)(const unsigned int));

};
#endif
