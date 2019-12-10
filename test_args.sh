#!/bin/bash


#arg1="--ls1 1 2 3" "--ls1 1 2 3 4 5 6"
#arg2="--ls2 13 14" "--ls2 13 14"


#langs="EO" "FI" "ID" "KO" "NO" "TR" "VI"
#argstogether=("--ls1 1 2 3 --ls2 4 5 6 7" "--ls1 13 14 --ls2 23 24")






langs=("EO" "FI" "ID" "KO" "NO" "TR" "VI")

argstogether=("--lang EO --factors 2 6 10 14 18 22 26 --hist_lens 2 4 8 16 32"
              "--lang FI --factors 2 6 10 14 18 22 --hist_lens 2 4 8 16 32 64 81"
              "--lang ID --factors 2 6 10 14 18 --hist_lens 2 4 8 16 32 64 81"
              "--lang KO --factors 2 6 10 14 --hist_lens 2 4 8 16 32 64 81"
              "--lang NO --factors 2 6 10 14 18 22 26 --hist_lens 2 4 8 16 32"
              "--lang TR --factors 2 --hist_lens 2 4 8 16 32 64 81"
              "--lang VI --factors 2 6 10 14 18 --hist_lens 2 4 8 16 32")

for i in $(seq 0 6); do 
    l="${langs[i]}"
    a="${argstogether[i]}" 
    echo "$l" "__" "$a"; 
    echo 

    python3 test_args.py $a

    echo
    echo "done!"
done


for a in "${argstogether[@]}"; do
    echo "let's try this"
    echo $a
    echo 

    python3 test_args.py $a

    echo
    echo "done!"
done