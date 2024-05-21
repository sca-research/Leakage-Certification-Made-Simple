typedef double (*knn_fns)(const int n_sample);


double knnf01(const int n) {return round( log(n) );}
double knnf02(const int n) {return round( pow( log(n), 2.0) );}
double knnf0(const int n) {return round( pow(log10(n), 2.0) );}

double knnf1(const int n) {return round( pow((double)n, 0.33)/(double)pow( log10(n), .25 ) + 0.5*log10(n) );}
double knnf2(const int n) {return round( pow((double)n, 0.33)/(double)pow( log10(n), .25 ) );}
double knnf3(const int n) {return round( pow((double)n, 0.33) );}
double knnf4(const int n) {return round( pow((double)n, 0.33)/(double)sqrt( log10(n) ) );}
double knnf5(const int n) {return round( pow((double)n, 0.33)/(double)log10(n) );}
double knnf6(const int n) {return round( pow((double)n, 0.4) );}
double knnf7(const int n) {return round( pow((double)n, 0.4)/(double)pow( log10(n), .25 ) );}

double knnf8(const int n) {return round( pow(log10(n), 2.0) + log10(n));}
