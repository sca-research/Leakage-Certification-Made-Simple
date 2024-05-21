#include <limits>
#include <filesystem>
#include <regex>
#include <stdarg.h>
#include <stdio.h>


// bool Verbse = false;


enum class experr
  {

   NOSIGMA = 701,
   UNKNWNPARAM,      // unknown experiment param
   NOTRCFILE,        // no filename to save traces
   NOINPUT,          // no input data
  };


struct exprmnt_param {
  double sigma;
  unsigned int n_data;
  std::string ifile;
  std::string ofile;
  std::string noise_distr_name;
  std::string model_name;
  bool no_sigma;
  
};

bool parse_cmd(int n, char* argv[], struct exprmnt_param *ep) {

  ep->sigma = -1000.0;   /* set to large negative number; indicates sigma to be read from file */
  ep->n_data = 15000;    /* by default 15000 traces will be simulated */
  ep->noise_distr_name = "gauss"; /* default noise distribution is Normal */
  ep->no_sigma = false;
  ep->model_name = "hw";
  
  char *stopstring;

  bool sflag = false, tflag = false, iflag = false;
  //ep.no_out = true;

  //use getopt in C to parse
  for(int i = 1;i < n;)
    {
      if(strcmp(argv[i], "-s") == 0)
	{
	  ep->sigma = strtod(argv[i+1], &stopstring);
	  sflag = true;
	  i += 2;
	}
      else if(strcmp(argv[i], "-t") == 0)
	{
	  ep->ifile = argv[i+1];
	  tflag = true;
	  i += 2;
	}
      else if(strcmp(argv[i], "-f") == 0)
	{
	  ep->ifile = argv[i+1];
	  iflag = true;
	  if(!std::filesystem::exists(ep->ifile))   /* requires using g++-8 with -lstdc++fs on ubuntu */
	    {
	      std::cout<<"Error: "<<ep->ifile<<" does not exist!\n";
	      return false;
	    }
	  i += 2;
	}
      else if(strcmp(argv[i], "-n") == 0)
	{
	  ep->n_data = atoi(argv[i+1]);
	  i += 2;
	}
      else if(strcmp(argv[i], "-o") == 0)
	{
	  // std::regex e("(-)(.*)");
	  // bool flag = false;
	  // while(i+1 < n)
	  //   {
	  //     i += 1;
	  //     if(std::regex_match(argv[i], e))
	  // 	{
	  // 	  if(ep.ofile.empty()) ep.no_out = true;
	  // 	  break;
	  // 	}
	  //     else
	  // 	{
	  // 	  ep.no_out = false;
	  // 	  ep.ofile.push_back(argv[i]);
	  // 	}

	  //   }
	  ep->ofile = argv[i+1];
	  i += 2; 
	}
      else if(strcmp(argv[i], "-nd") == 0)
	{
	  ep->noise_distr_name = argv[i+1];
	  i += 2;
	}
      else if(strcmp(argv[i], "-ns") == 0)
	{
	  ep->no_sigma = (atoi(argv[i+1]) > 0)? true: false;
	  i += 2;
	}
      else if(strcmp(argv[i], "-m") == 0)
	{
	  ep->model_name = argv[i+1];
	  i += 2;
	}
      else
	{
	  printf("Undefined option!\n");
	  return false;
	}
    }

  if(tflag && !sflag)
    {
      printf("No sigma value given\n");
      return false;
    }
  if(sflag && !tflag)
    {
      printf("No file name given for traces!\n");
      return false;
    }
  if(!tflag && !iflag)
    {
      printf("No input data file given\n");
      return false;
    }

   if(ep->ofile.empty())
     printf("No output file given\n");
  
  return true;

}


// int verbose_print(const char * restrict format, ...) {
  
//   if( !pflag )
//     return 0;
  
//   va_list args;
//   va_start(args, format);
//   int ret = vprintf(format, args);
//   va_end(args);
  
//   return ret;
// }
