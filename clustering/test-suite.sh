#!/bin/bash
clusters="3,4,5"

for k in `seq 3 5`;
do
    for i in `seq 1 $2`;
    do
        echo "Testing for k = $k with $1 algorithm : $i"
        filename="$k-mean-$1-$i"
        ./cluster.py -k $k --algorithm=$1 --filename=$filename > "output/$filename.log"
    done

done
