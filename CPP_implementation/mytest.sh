#g++ -std=c++17 -DNTRIAL=1 -O2 main.cpp mi_estimator.cpp simulate_trace.cpp -lmlpack -fopenmp -larmadillo -lgsl -lstdc++fs 

#GREEN='\033[0;32m'
#NC='\033[0m'

ifnames=( pp_msk )
#ifnames=( tr )



ofnames=( pp_out)
#sval=( 0.05 0.5 1 )

#nf=${#ifnames[@]}

#for str in ${fnames[@]}; do
for ((i= 1  ; i<= 276; i++))
do
   
    istr=${ifnames[0]}$i".csv"
    ostr=${ofnames[0]}$".csv" 
    printf "Running experiment for input %s, output in -> %s\n" $istr ${sval[$i]} $ostr
    #./test -s ${sval[$i]} -t $istr -o $ostr
    ./a.out -ns 1 -f PP_MSK_/$istr -o $ostr
done
