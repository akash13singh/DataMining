#!/bin/bash
clusters="3,4,5"

for k in `seq 3 5`;
do
    echo "Testing for k = $k with $1 algorithm"
    ./cluster.py -k $k --algorithm=$1 > "output/$k-mean-$1.log"
done
