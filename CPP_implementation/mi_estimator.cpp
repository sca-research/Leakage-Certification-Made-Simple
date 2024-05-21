#include "mi_estimator.hpp"
#include <chrono>
#include <omp.h>
#include <assert.h>
#include <algorithm>
/*
--------------------------
On Apple Clang, you need to add several options to use OpenMP's front end
instead of the standard driver option. This usually looks like
  -Xpreprocessor -fopenmp -lomp
------------------------------
 */


arma::uvec sample_wor(int n_data, int n_sample) {

  std::random_device rd;
  std::mt19937 gen(rd()); //seed the generator
  std::uniform_int_distribution<int> distr(0, n_data-1);
  
  arma::uvec data_index(n_sample, arma::fill::zeros);
  
  int i = 0;
  while(i < n_sample)
    {
      if(data_index(i) == 0)
	{
	  data_index(i) = distr(gen);
	  i++;
	} 
    }

  //for(int i = 0; i < n_sample;i++) data_index(i) = distr(gen);      
  return data_index;
  
}

arma::vec
mi_estimator::mi_multiple_sample(const int k) {

  int n_points = data.n_cols-1;
  int n_sample = data.n_rows;
  arma::uvec col_index(2, arma::fill::zeros);
  arma::vec result(n_points, arma::fill::zeros);
  
  for(int ipoint = 0;ipoint < n_points;ipoint++)
    {
      
      col_index = (0,(unsigned int)ipoint+1);
      arma::mat data_point = data.cols(col_index);
      NeighborSearch<NearestNeighborSort, mydistMetric, arma::mat, tree::VPTree> nn(data_point.t());

      arma::Mat<size_t> neighbors;        // Create the object to store the nearest neighbors in.
      arma::mat distances;                // store the distance too.           
      nn.Search(k, neighbors, distances); // Compute the neighbors.

      double sum = 0, r, maxk;
      int nx, ny;
      for(int i = 0;i < n_sample;i++)
	{
	  
	  r = (double)distances(k-1,i);	  
	  maxk = k;
	  if(r == 0)
	    {
	      int count = 0;;
	      for(int j = 0;j < n_sample;j++)
		{
		  double d = arma::norm(data_point.row(i)-data_point.row(j), "inf");
		  if(d == 0) count++;
		}
	      maxk = (double) count;
	    }
	  
	  nx = 0;ny = 0;
	  //int j;\
	  //omp_set_num_threads(4);
          //#pragma omp parallel  
	  //{
          //#pragma omp for reduction(+:nx, ny)
	  for(int j = 0;j < n_sample;j++)
	    {	    
	      if(j != i)
	  	{
	  	  if(abs(data_point(j, 0) - data_point(i, 0)) <= r) nx += 1;
	  	  if(abs(data_point(j, 1) - data_point(i, 1)) <= r) ny += 1;
	  	}	      
	    }
	  //}

	  sum = sum + boost::math::digamma(maxk);
	  sum = sum - log((double)(nx + 1));
	  sum = sum - log((double)(ny + 1));

	}
      
      sum = sum/(double)n_sample;
      sum = sum + log((double)n_sample);

      result(ipoint) = sum;

    }//n_points for loop

  return result;

}


struct mi_t
mi_estimator::mi_mixed(const int n_sample, const int k, const int n_trial) {

  struct mi_t result; /* will contain sample mean and variance of mi */
  
  if(data.n_rows < n_sample)
    {
      std::cout<< "Error: sample size must be more than data size!!\n";
      result.mi_avg = -1;
      result.mi_var = -1;
      return result;
    }
  
  arma::vec mi_vec(n_trial, arma::fill::zeros);
  
  for(int itrial = 0;itrial < n_trial;itrial++)
    {

      arma::uvec index = sample_wor((int) data.n_rows, n_sample);
      arma::mat sampled_data = data.rows(index);
      
      //NeighborSearch<NearestNeighborSort, mydistMetric, arma::mat, tree::VPTree> nn(sampled_data.t());
      NeighborSearch<NearestNeighborSort, mydistMetric, arma::mat, tree::BallTree> nn(sampled_data.t());
      
      arma::Mat<size_t> neighbors;        // Create the object to store the nearest neighbors in.
      arma::mat distances;                // store the distance too.           
      nn.Search(k + 1, neighbors, distances); // Compute the neighbors.


      //auto start = std::chrono::high_resolution_clock::now();
      double sum = 0, r;
      int nx, ny;
      for(int i = 0;i < n_sample;i++)
	{
	  
	  r = (double)distances( k , i);  // Need to look at this parameter	  
	  double maxk = k;
	  if(r == 0)
	    {
	      int count = 0;;
	      for(int j = 0;j < n_sample;j++)
		{
		  double d = arma::norm(sampled_data.row(i)-sampled_data.row(j), "inf");
		  if(d == 0) count++;
		}
	      maxk = (double) count;
	    }
	  
	  nx = 0;ny = 0;
	  int j;
	  
	  // omp_set_num_threads(8);
          #pragma omp parallel  
	  {
          #pragma omp for reduction(+:nx, ny)
	  for(j = 0;j < n_sample;j++)
	    {	    
	      if(j != i)
	  	{
	  	  if(abs(sampled_data(j, 0) - sampled_data(i, 0)) <= r) nx += 1;
	  	  if(abs(sampled_data(j, 1) - sampled_data(i, 1)) <= r) ny += 1; /* comment this for bivriate data */
		 /*For bi-variate trace data (i.e. Y), please consider the following*/ 
	  	  //if(std::max( abs(sampled_data(j, 1) - sampled_data(i, 1)),  abs(sampled_data(j, 2) - sampled_data(i, 2))) <= r ) ny += 1;

	  	}	      
	    }
	  }
	  //printf("maxk = %lf, digama = %lf\n", maxk, boost::math::digamma(maxk));
	  sum = sum + boost::math::digamma(maxk);
	  sum = sum - log((double)(nx + 1));
	  sum = sum - log((double)(ny + 1));

	}
      
      sum = sum/(double)n_sample;
      sum = sum + log((double)n_sample);
      
      mi_vec(itrial) = sum * 1.4426950408889634 ;
     
      std::cout<<"\rIteration = "<<itrial+1;
      fflush(stdout);
      //auto stop = std::chrono::high_resolution_clock::now();

      //std::chrono::duration<double> fp_ms = stop - start;
  
      //std::cout << "Time taken: "
      // 		<< fp_ms.count() << " seconds" << std::endl;
    }
  
  result.mi_avg = arma::mean(mi_vec);
  result.mi_var = arma::var(mi_vec);
  printf("\n"); 
  printf("Estimate variance    = % .18f\n", result.mi_var);

  return result;
  
}


struct mi_t
mi_estimator::mi_ksg(const int n_sample, const int k, const int n_trial) {

  struct mi_t result; /* will contain sample mean and variance of mi */
  
  if(data.n_rows < n_sample)
    {
      std::cout<< "Error: sample size must be more than data size!!\n";
      result.mi_avg = -1;
      result.mi_var = -1;
      return result;
    }
  
  arma::vec mi_vec(n_trial, arma::fill::zeros);

  double logterm = log((double)n_sample);  
  for(int itrial = 0;itrial < n_trial;itrial++)
    {

      arma::uvec index = sample_wor((int) data.n_rows, n_sample);
      arma::mat sampled_data = data.rows(index);
      
      NeighborSearch<NearestNeighborSort, mydistMetric, arma::mat, tree::VPTree> nn(sampled_data.t());
      // Create the object to store the nearest neighbors in.
      arma::Mat<size_t> neighbors;
      arma::mat distances; // store the distance too.
      
      // Compute the neighbors.     
      nn.Search(k, neighbors, distances);
      
      double sum = 0, r;
      int nx, ny;
      for(int i = 0;i < n_sample;i++)
	{
	  r = (double)distances(k-1,i);
	  double maxk = k;
	  nx = 0;ny = 0;
	  for(int j = 0;j < n_sample;j++)
	    {	    
	      if(j != i)
		{
		  if(abs(sampled_data(j, 0) - sampled_data(i, 0)) <= r) nx += 1;
		  if(abs(sampled_data(j, 1) - sampled_data(i, 1)) <= r) ny += 1;
		}
	      
	    }
	  //std::cout<<"Num of threads "<<omp_get_num_threads()<<"\n";
	  //printf("maxk = %lf, digama = %lf\n", maxk, boost::math::digamma(maxk));
	  /*int dummyx = 0, dummyy = 0;
	  for(int j = 0;j < n_sample;j++)
	    {
	      if(j != i)
		{
		  if(abs(sampled_data(j, 0) - sampled_data(i, 0)) <= r) dummyx += 1;
		  if(abs(sampled_data(j, 1) - sampled_data(i, 1)) <= r) dummyy += 1;
		}
	    }
	  assert(nx == dummyx);
	  assert(ny == dummyy);*/
	  
	  sum = sum + boost::math::digamma(maxk) - boost::math::digamma((double)(nx + 1)) - boost::math::digamma((double)(ny + 1));
	}

      //auto stop = std::chrono::high_resolution_clock::now();
      //std::chrono::duration<double> tt = stop-start;
      //std::cout<<"Time taken  "<<tt.count()<<" seconds\n";

      
      sum = sum/(double)n_sample;
      sum = sum + logterm;
      
      mi_vec(itrial) = sum;
     
      //printf("\rIteration = %d", itrial+1);
      //fflush(stdout);
    }
  
  result.mi_avg = arma::mean(mi_vec);
  result.mi_var = arma::var(mi_vec);
  printf("\n"); 
  printf("Estimate variance    = % .18f\n", result.mi_var);
  printf("Bias     = %0.9f\n", pow(log(n_sample), 3)/sqrt(n_sample));
  return result;
}



struct mi_t
mi_estimator::mi_truncksg(const int n_sample, const int k, const int n_trial) {

  struct mi_t result; /* will contain sample mean and variance of mi */
  
  if(data.n_rows < n_sample)
    {
      std::cout<< "Error: sample size must be more than data size!!\n";
      result.mi_avg = -1;
      result.mi_var = -1;
      return result;
    }
  
  arma::vec mi_vec(n_trial, arma::fill::zeros);
  double threshold = pow(log(n_sample),0.5)/sqrt(n_sample);

  double logterm = log(n_sample);
  
  for(int itrial = 0;itrial < n_trial;itrial++)
    {

      arma::uvec index = sample_wor((int) data.n_rows, n_sample);
      arma::mat sampled_data = data.rows(index);
      
      NeighborSearch<NearestNeighborSort, mydistMetric, arma::mat, tree::VPTree> nn(sampled_data.t());
      
      arma::Mat<size_t> neighbors;
      arma::mat distances; 
      
      
      // Compute the neighbors.     
      nn.Search(k, neighbors, distances);
      
      double sum = 0, r, tsum = 0;
      int nx, ny;
      for(int i = 0;i < n_sample;i++)
	{
	  
	  r = (double)distances(k-1,i);
	  double maxk = k;
	  
	  nx = 0;ny = 0;
	  for(int j = 0;j < n_sample;j++)
	    {	    
	      if(j != i)
		{
		  if(abs(sampled_data(j, 0) - sampled_data(i, 0)) <= r) nx += 1;
		  if(abs(sampled_data(j, 1) - sampled_data(i, 1)) <= r) ny += 1;
		}
	      
	    }  
	  //printf("maxk = %lf, digama = %lf\n", maxk, boost::math::digamma(maxk));
	  tsum = boost::math::digamma(maxk) - boost::math::digamma((double)(nx + 1)) - boost::math::digamma((double)(ny + 1));
	  tsum += logterm;
	  
	  if(r <= threshold)
	    sum += tsum;
	}
      
      sum = sum/(double)n_sample;
      //sum = sum + log((double)n_sample);
      
      mi_vec(itrial) = sum;
     
      std::cout<<"\rIteration = "<<itrial+1;
      fflush(stdout);
    }
  
  result.mi_avg = arma::mean(mi_vec);
  result.mi_var = arma::var(mi_vec);
  printf("\n"); 
  printf("Estimate variance    = % .18f\n", result.mi_var);

  return result;
  
}


struct mi_t
mi_estimator::mi_biksg(const int n_sample, const int k, const int n_trial) {

  struct mi_t result; /* will contain sample mean and variance of mi */
  
  if(data.n_rows < n_sample)
    {
      std::cout<< "Error: sample size must be more than data size!!\n";
      result.mi_avg = -1;
      result.mi_var = -1;
      return result;
    }
  
  arma::vec mi_vec(n_trial, arma::fill::zeros);
  
  for(int itrial = 0;itrial < n_trial;itrial++)
    {

      arma::uvec index = sample_wor((int) data.n_rows, n_sample);
      arma::mat sampled_data = data.rows(index);
      
      NeighborSearch<NearestNeighborSort, EuclideanDistance, arma::mat, tree::VPTree> nn(sampled_data.t());
      // Create the object to store the nearest neighbors in.
      arma::Mat<size_t> neighbors;
      arma::mat distances; // store the distance too.
      
      
      // Compute the neighbors.     
      nn.Search(k, neighbors, distances);
      
      double sum = 0, r, d;
      int nx, ny;
      for(int i = 0;i < n_sample;i++)
	{
	  
	  r = (double)distances(k-1,i);   /* distance of the k th nearest neighbour */
	  double maxk = k;
	  if(r == 0)
	    {
	      int count = 0; 
	      for(int j = 0;j < n_sample;j++)
		{
		  d = arma::norm(sampled_data.row(i)-sampled_data.row(j), 2);
		  if(d == 0) count++;
		}
	      maxk = (double) count;
	    }
	  
	  nx = 0;ny = 0;
	  for(int j = 0;j < n_sample;j++)
	    {	    
	      if(j != i)
		{
		  if(abs(sampled_data(j, 0) - sampled_data(i, 0)) <= r) nx += 1;
		  if(abs(sampled_data(j, 1) - sampled_data(i, 1)) <= r) ny += 1;
		}
	      
	    }  
	  //printf("maxk = %lf, digama = %lf\n", maxk, boost::math::digamma(maxk));
	  sum = sum + boost::math::digamma(maxk);
	  sum = sum - log((double)nx);
	  sum = sum - log((double)ny);
	}
      
      sum = sum/(double)n_sample;
      sum = sum + log((double)n_sample);
      double gn = sqrt(M_PI)/boost::math::tgamma(1.5);
      gn *= gn;
      double gd = M_PI/boost::math::tgamma(2);
      sum = sum + log(gn/gd);
      
      mi_vec(itrial) = sum;
     
      printf("\rIteration = %d", itrial+1);
      fflush(stdout);
    }
  
  result.mi_avg = arma::mean(mi_vec);
  result.mi_var = arma::var(mi_vec);
  printf("\n"); 
  printf("Estimate variance    = %.9f\n", result.mi_var);

  return result;
  
}




