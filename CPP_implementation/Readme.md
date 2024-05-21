This folder includes the CPP implementation of nearest neighbor-based MI estimation(GKOV).

### Dependencies
- mlpack (and corresponding dependencies e.g. armadillo, boost)
- gsl

### Compile options

- On OSX
`g++ -std=c++17 -O2 main.cpp simulate_trace.cpp mi_estimator.cpp [linking path] -larmadillo -lmlpack -lgsl -fopenmp`


- On ubuntu

`g++-8 -std=c++17 -O2 main.cpp simulate_trace.cpp mi_estimator.cpp [linking path] -larmadillo -lmlpack -lgsl -lstdc++fs -fopenmp`

Use of linking path is optional depending on your system configuration

### How to use 

- `./a.out -s 3 -t [infilename].csv -o [outfilename].csv`

This will generate a .csv trace file using sigma value as 3. It will compute the MI (for different sample sizes) from the input in [infilename].csv. The output will be stored in the [outfilename].csv. If you want to generate a specified number of traces then use the option `-n` (e.g. `-n 50000`). The deault option is 15000.  

- If you already have a .csv file then you can use it computed MI (for different sample sample size) using

`./a.out -f [infilename].csv -o [outfilename].csv`
We already provided two input datasets for the trial run: `tr_hw_1.5.csv` and `tr_hw_4G.csv`. If no output filename is given then outputs are currently printed on terminal.

- Note that the .csv trace file generated will contain secret key and sigma values on the last line. This value is used to compute the exact MI using numerical integration method from GSL. If your file do not have the sigma value or you do not wish to compute the exact MI then use an additional option `-ns 1`




