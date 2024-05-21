#include <gsl/gsl_integration.h>
#include <gsl/gsl_randist.h>
#include <boost/math/special_functions/binomial.hpp>

double gmf(double x, void *params);
double exact_mi_hw_normal(double a, double b, int n, double (*func)(double, void*), void *param);

double trapezoidal_rule(double, double, int, double (*func)(double, void*), void *);

double exact_mi_hw_normal(double a, double b, int n, double (*func)(double, void*), void *param) {


  gsl_integration_workspace * w
    = gsl_integration_workspace_alloc (n);
  double result, error;
  
  gsl_function F;
  F.function = func;
  F.params = param;

  gsl_integration_qag (&F, a, b, 0, 1e-6, n, GSL_INTEG_GAUSS41,
                        w, &result, &error);
  gsl_integration_workspace_free (w);

  // printf ("result (GSL QAG)     = % .18f\n", result);

  // double r = trapezoidal_rule(a, b, 1000, func, param);
  // printf ("result (TRAPZ)       = % .18lf\n", r);

  //printf("infty = %lf\n", std::numeric_limits<double>::infinity());

  double s = *((double *)param + 1);

  double normal_entrpy = 2*M_PI*exp(1.0);
  normal_entrpy *= (s*s);
  normal_entrpy = log2(normal_entrpy);
  normal_entrpy /= 2.0;

  normal_entrpy = result - normal_entrpy;
  //printf("MI     (GSL)         = % .18f\n", (result - constval));
  return normal_entrpy;

}


double gmf(double x, void *params) {

  double n = *(double *)params;
  double s = *((double *)params + 1);

  double t, sum = 0.0, p;
  for(int i = 0; i <= (int)n;i++)
    {
      // t = x - (double) i;
      // p = t*t; p = p/(s*s);
      // p = exp(-p/2);
      // p = p/(s*sqrt(2*M_PI));
      // t = boost::math::binomial_coefficient<double>((int)n, i);
      // t /= pow(2.0, (double)n);
      // sum += (p*t);
      t = x - (double) i;
      p = gsl_ran_gaussian_pdf(t, s);
      t = boost::math::binomial_coefficient<double>((int)n, i);
      t /= pow(2.0, (double)n);
      sum += (p*t);
    
    }

  return -sum*log2(sum);

}

double trapezoidal_rule(double a, double b, int n, double (*func)(double, void*), void *param) {
  
  double trapez_sum;
  double fa, fb, x, step;
  int j;
  step = (b-a)/((double) n);
  fa = (*func)(a, param)/2. ; fb = (*func)(b, param)/2. ;

  trapez_sum = 0.0;
  
  for (j=1; j <= n-1; j++)
    {
      x = j*step+a;
      trapez_sum += (*func)(x, param);
    }
  trapez_sum = (trapez_sum + fb + fa)*step;
  return trapez_sum;
}




    
  


  


 


