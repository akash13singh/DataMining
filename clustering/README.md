## Command
```
$ ./cluster.py  -h
Usage: cluster.py [options]

Options:
  -h, --help            show this help message and exit
  -a ALGORITHM, --algorithm=naive|...
                        Algorithm for initialization center points
  -k K, --cluster=K
```

# Example
```
$ ./cluster.py -k 5
```

# Log pattern
```
k: 2
algorithm: naive
loss-function: mean
---
iteration 1 : [12.886419123241014, -2.7210855266185212][ 22.0108 -10.9737]
---
0:[-19.0748  -8.536 ]:0
1:[ 22.0108 -10.9737]:0
.
.
.
---
total_cost: 18448.440155
iteration: 2
```
