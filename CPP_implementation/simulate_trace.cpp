#include "simulate_trace.hpp"

bool
trace::trace1d(const uint32_t n_trc, const unsigned int secret_key,
	       std::string distr_name, const double sigma, unsigned int (*lkg_func)(const unsigned int)) {
  
  std::random_device rd;
  std::mt19937 gen(rd()); //seed the generator
  
  int ub = (1UL<<20)-1;
  std::uniform_int_distribution<int> udistr(0, ub);
  
  std::default_random_engine generator(rd());
   
  FILE *ofp;
    
  if(distr_name == "gauss")
    {
      std::normal_distribution<double> gd(0.0, sigma);
      uint8_t u;
      int i;
      ofp = fopen(fname.c_str(), "w");
      if(ofp == NULL)
	{
	  std::cout<< "Can not open "<<fname <<"\n";
	  return false;
	}
      for(i = 0;i < n_trc;i++)
	{
	  u = udistr(gen)&0xff;
	  int z, y;
	  y = crypto_func(u, secret_key);
	  z = lkg_func(y);
	  double tr = (double)z + gd(generator); //trc value = model(sbox) + noise 
	  // write to a file.	
	  fprintf(ofp, "%d, %lf\n", u, tr);
	}
    }
  else if(distr_name == "laplace")
    {
      
      boost::random::laplace_distribution lpd((double)0,sigma); /** first argument must be double or 0.0 (when = 0), otherwise boost raises error */
      uint8_t u;
      int i;
      ofp = fopen(fname.c_str(), "w");
      if(ofp == NULL)
	{
	  std::cout<< "Can not open "<<fname <<"\n";
	  return false;
	}
      for(i = 0;i < n_trc;i++)
	{
	  u = udistr(gen)&0xff;
	  int z, y;
	  y = crypto_func(u, secret_key);
	  z = lkg_func( y );
	  double tr = (double)z + lpd(generator); 
	  // write to a file.
	  fprintf(ofp, "%d, %lf\n", y, tr);
	}
    }
  else if(distr_name == "lognormal")
    {
      
      std::lognormal_distribution<double> lgn(0,sigma);
      uint8_t u;
      int i;
      ofp = fopen(fname.c_str(), "w");
      if(ofp == NULL)
	{
	  std::cout<< "Can not open "<<fname <<"\n";
	  return false;
	}
      for(i = 0;i < n_trc;i++)
	{
	  u = udistr(gen)&0xff;
	  int z, y;
	  y = crypto_func(u, secret_key);
	  z = lkg_func(y); 
	  double tr = (double)z + lgn(generator); 
	  // write to a file.
	  fprintf(ofp, "%d, %lf\n", y, tr);
	}
    }
  else
    {
      std::cout<< "Unknown noise distribution!\n";
      return false;
    }

  
  fprintf(ofp, "%d, %lf\n", secret_key, sigma); /* last row stores the secret and sigma. Ignore it for analysis */
  fclose(ofp);
  
  return true;
  
}

bool
trace::trace2d_mask(const uint32_t n_trc, const unsigned int secret_key,
		      std::string distr_name, const double sigma, unsigned int (*lkg_func)(const unsigned int)) {


  std::random_device rd;
  std::mt19937 gen(rd()); //seed the generator
  
  int ub = (1UL<<20)-1;
  std::uniform_int_distribution<int> udistr(0, ub);
  
  std::default_random_engine generator(rd());
    
  FILE *ofp; 
  ofp = fopen(fname.c_str(), "w");
  if(ofp == NULL)
    {
      printf("Can not open file %s\n", fname.c_str());
      return false;
    }
  
  std::normal_distribution<double> gd(0.0, sigma);

  uint8_t u, mask;
  double x, x1;
  for(int i = 0;i < n_trc;i++)
    {
      u = udistr(gen)&0xff;
      int z, y;
      y = crypto_func(u, secret_key);
      mask = udistr(gen)&0xff;
      z = lkg_func(y^mask);
      x  = (double)z + gd(generator);
      x1 = (double)lkg_func(mask) + gd(generator);
      // write to a file.
      fprintf(ofp, "%d, %lf, %lf\n", y, x, x1 );
    }

 
  fprintf(ofp, "%d, %lf, %lf\n", secret_key, sigma, sigma); /* last row stores the secret and sigma. Ignore it for analysis */
  fclose(ofp);
  
  return true;
 
}

bool
trace::trace1hd(const uint32_t n_trc, const unsigned int secret_key,
	       std::string distr_name, const double sigma,
	       unsigned int (*lkg_func)(const unsigned int, const unsigned int)) {
  
  std::random_device rd;
  std::mt19937 gen(rd()); //seed the generator
  
  int ub = (1UL<<20)-1;
  std::uniform_int_distribution<int> udistr(0, ub);
  
  std::default_random_engine generator(rd());
   
  FILE *ofp;
  ofp = fopen(fname.c_str(), "w");
  if(ofp == NULL)
    {
      std::cout<< "Can not open "<<fname <<"\n";
      return false;
    }
      
  if(distr_name == "gauss")
    {
      std::normal_distribution<double> gd(0.0, sigma);
      uint8_t u;
      int i;
      ofp = fopen(fname.c_str(), "w");
      if(ofp == NULL)
	{
	  std::cout<< "Can not open "<<fname <<"\n";
	  return false;
	}
      for(i = 0;i < n_trc;i++)
	{
	  u = udistr(gen)&0xff;
	  int z, y;
	  y = crypto_func(u, secret_key);
	  z = lkg_func(y, u^secret_key);
	  double tr = (double)z + gd(generator); //trc value = model(sbox) + noise 
	  // write to a file.
	  fprintf(ofp, "%d, %lf\n", y, tr);
	  
	}
    }
  else if(distr_name == "laplace")
    {
      
      boost::random::laplace_distribution lpd((double)0,sigma); /** first argument must be double or 0.0 (when = 0), otherwise boost raises error */
      uint8_t u;
      int i;
      ofp = fopen(fname.c_str(), "w");
      for(i = 0;i < n_trc;i++)
	{
	  u = udistr(gen)&0xff;
	  int z, y;
	  y = crypto_func(u, secret_key);
	  z = lkg_func(y, u^secret_key);
	  double tr = (double)z + lpd(generator); 
	  // write to a file.
	  fprintf(ofp, "%d, %lf\n", y, tr);
	}
    }
  else if(distr_name == "lognormal")
    {
      
      std::lognormal_distribution<double> lgn(0,sigma);
      uint8_t u;
      int i;
      ofp = fopen(fname.c_str(), "w");
      for(i = 0;i < n_trc;i++)
	{
	  u = udistr(gen)&0xff;
	  int z, y;
	  y = crypto_func(u, secret_key);
	  z = lkg_func(y, u^secret_key);
	  double tr = (double)z + lgn(generator); 
	  // write to a file.
	  fprintf(ofp, "%d, %lf\n", y, tr);
	}
    }
  else
    {
      std::cout<< "Unknown noise distribution!\n";
      return false;
    }
  
  fprintf(ofp, "%d, %lf\n", secret_key, sigma); /* last row stores the secret and sigma. Ignore it for analysis */
  fclose(ofp);
  
  return true;
  
}

bool
trace::trace1d_mask(const uint32_t n_trc, const unsigned int secret_key,
		    std::string distr_name, const double sigma,
		    unsigned int (*lkg_func)(const unsigned int)) {
  

  std::random_device rd;
  std::mt19937 gen(rd()); //seed the generator
  
  int ub = (1UL<<20)-1;
  std::uniform_int_distribution<int> udistr(0, ub);
  
  std::default_random_engine generator(rd());
  
  FILE *ofp; 
  ofp = fopen(fname.c_str(), "w");
  if(ofp == NULL)
    {
      printf("Can not open file %s", fname.c_str());
      return false;
    }
  
  std::normal_distribution<double> gd(0.0, sigma);

  uint8_t u, mask;
  double x, x1;
  for(int i = 0;i < n_trc;i++)
    {
      u = udistr(gen)&0xff;
      int z, y;
      mask = udistr(gen)&0xff;
      y = crypto_func(u, secret_key);
      z = lkg_func(y^mask);
      double tr  = (double)z + (double)lkg_func(mask) + gd(generator);
      
      fprintf(ofp, "%d, %lf\n", y, tr); // write to a file.
    }

 
  fprintf(ofp, "%d, %lf\n", secret_key, sigma); /* last row stores the secret and sigma. Ignore it for analysis */
  fclose(ofp);
  
  return true;
 
}



unsigned int hamming_wt(const unsigned int a) {
    return __builtin_popcount(a);
}

unsigned int identy(const unsigned int a) {
    return a;
}

unsigned int nonlinear_sb(const unsigned int a) {
  /*ARIA Sbox S1*/
  return sb1[a];
}

unsigned int square(const unsigned int a) {
  return a*a;
}
unsigned int whamming_wt(const unsigned int a) {
  uint8_t wt = 0x71; // wt can be changed here
  return __builtin_popcount(a&wt);
}

unsigned int hamming_dist(const unsigned int a, const unsigned int b) {
  return __builtin_popcount(a^b);
}

unsigned int sbox(const unsigned int x, const unsigned int key) {
  return Sbox[x^key];
}


