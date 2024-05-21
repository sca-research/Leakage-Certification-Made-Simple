//#include <armadillo>
#include "numerical_comp.hpp"
#include "simulate_trace.hpp"
#include "mi_estimator.hpp"
#include <algorithm>

#include "util.hpp"
#include "knn_func.hpp"


#include <stdio.h>
//#include <gsl/gsl_sf_psi.h>

#ifndef NTRIAL
#define NTRIAL 10
#endif


bool generate_trace(unsigned int skey, struct exprmnt_param ep);

double exact_mi(struct exprmnt_param ep);

int main(int argc, char* argv[]) {
 	

struct exprmnt_param ep;
  unsigned int skey = 137;
  bool oflag = true;

  bool e;
  if(argc > 1)
    e = parse_cmd(argc, argv, &ep);
  else
    {
      std::cout<< "No input given\n";
      return -1;
    }
  if(!e) return -1;

  // if(ep.sigma > 0 && !ep.ifile.empty())
  //  {
  //     if( !generate_trace(skey, ep) )
	// {
  //     		std::cout<< "Trace generation failed.\n";
	//   return -1;
	// }
  //   }
  if(ep.sigma > 0 && !ep.ifile.empty())
    {
      trace T(ep.ifile, ep.model_name, &sbox);
      printf("Simulating %d traces with %s distribution, sigma = %lf\n", ep.n_data, ep.noise_distr_name.c_str(), ep.sigma);
      // T.trace2d_mask(ep.n_data, skey, ep.noise_distr_name, ep.sigma, &hamming_wt);  /* For masked data */
      // T.trace1d_mask(ep.n_data, skey, ep.noise_distr_name, ep.sigma, &hamming_wt);
      T.trace1hd(ep.n_data, skey, ep.noise_distr_name, ep.sigma, &hamming_dist);

    }
  
  mi_estimator M(ep.ifile, ep.no_sigma);
  std::cout<< "Trace file "<< ep.ifile <<"\n";

  if(!ep.no_sigma && ep.sigma < 0)
    ep.sigma = M.sigma_from_file();
    
  if(ep.no_sigma || ep.sigma <= 0)
    std::cout<< "No (valid) sigma (skip exact MI computation)\n";

  double e_mi;
  /* Compute exact MI */
  if(ep.sigma > 0)
    e_mi = exact_mi(ep);
    
  int samples[] = {20000, 50000, 100000, 200000, 300000, 
       400000, 500000, 600000, 800000, 1000000};
 
  //int samples[] = {999999};
  // int samples[]= {100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 999999};
  int n_samples = sizeof(samples)/sizeof(samples[0]);
  int ntrial = NTRIAL;

  int k;

  knn_fns kf[] = {knnf01}; 
  //knn_fns kf[] = {knnf01, knnf0}; /* Different Knn functions: see knn_func.hpp */
  
  int n_kfs = sizeof(kf)/sizeof(kf[0]);

  FILE *ofp;
  if(!ep.ofile.empty())
    {
      if( std::filesystem::exists(ep.ofile) )
	ofp = fopen(ep.ofile.c_str(), "a");
      else
	ofp = fopen(ep.ofile.c_str(), "w");
      
      if(ofp == NULL)
	{
	  printf("Can not open output file %s\n", ep.ofile.c_str());
	  return -1;
	}
    }

#ifdef FIXEDK
  printf("K value is fixed\n");
  int kn[] = {7};
  n_kfs = sizeof(kn)/sizeof(kn[0]);
#endif

  for(int i = 0;i < n_samples;i++)
    {

      std::cout<< "sample size "<< samples[i]<< "\n";
      
      if(!ep.ofile.empty())
	fprintf(ofp, "%d", samples[i]);
      
      for(int j = 0; j < n_kfs;j++)
	{
  	  
	  struct mi_t ii;
  
#ifdef FIXEDK
	  
	  k = kn[j];
	  printf("K = %d\n", k);
	  ii = M.mi_truncksg(samples[i], (int)k, ntrial);
#else
	  
	  std::cout<< "Computing for knn function "<<j+1 <<"\n";
	  k = kf[j](samples[i]);
	  //k = 5;
	  printf("K = %d\n", k);
	  if(k < 3) k = 3;

	  ii = M.mi_mixed(samples[i], (int)k, ntrial);

#endif
	  
	  printf("Estimated MI         = % .18f\n", ii.mi_avg);
  	    
	  if(!ep.ofile.empty())
	    {
#ifdef FIXEDK
	  
	      fprintf(ofp, ", %lf, %lf, %d", ii.mi_avg, ii.mi_var, k);
#else
	      fprintf(ofp, ", %lf, %lf", ii.mi_avg, ii.mi_var);
#endif
	    }
	  //fprintf(ofp, ", %lf, %lf", ii.mi_avg, ii.mi_var);
	  
	}
      if(!ep.ofile.empty())
	{
	  if(!ep.no_sigma) fprintf(ofp, ", %lf\n", e_mi);
	  else
	    fprintf(ofp, "\n");
	}
    }

  if(!ep.ofile.empty())
    fclose(ofp);

  printf("MI     (GSL)         = % .9f\n", e_mi);
 
  return 0;
}


bool generate_trace(unsigned int skey, struct exprmnt_param ep) {

  trace T(ep.ifile, ep.model_name, &sbox);
  bool t = false;
  printf("Simulating %d traces with %s distribution, sigma = %lf\n", ep.n_data, ep.noise_distr_name.c_str(), ep.sigma);
  std::cout<< "Model used " << ep.model_name << "\n";

  if(ep.model_name == "hw")
    t = T.trace1d(ep.n_data, skey, ep.noise_distr_name, ep.sigma, &hamming_wt);
  else if(ep.model_name == "whw")
    t = T.trace1d(ep.n_data, skey, ep.noise_distr_name, ep.sigma, &whamming_wt);
  else if(ep.model_name == "id")
    t = T.trace1d(ep.n_data, skey, ep.noise_distr_name, ep.sigma, &identy);
  else if(ep.model_name == "hd")
    t = T.trace1hd(ep.n_data, skey, ep.noise_distr_name, ep.sigma, &hamming_dist);
    
  return t;
}

double exact_mi(struct exprmnt_param ep) {

  printf("Computing exact MI for sigma = %lf, ", ep.sigma);
  //Set parameters for exact MI computation  
  double e_mi = 0;
  if(ep.noise_distr_name == "gauss" && ep.model_name == "hw")
    {
      double alpha[2], a, b;
      alpha[0] = 8;  /* a parameter value to Gaussian mixture, set to 8 due to HW model of 8-bit Sbox*/ 
      alpha[1] = ep.sigma;
      a = -6*alpha[1];
      b = 8.0 + 6*alpha[1];
      
      e_mi = exact_mi_hw_normal(a, b, 1000, &gmf, alpha);
      printf("MI  : %lf\n", e_mi);
    }
  else
    printf("No exact MI support for this option\n");

  return e_mi;
    
}


