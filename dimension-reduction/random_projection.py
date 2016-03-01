#!/usr/local/bin/python
from optparse import OptionParser

import mnist_dataloader as data_loader
import math
import numpy as np
# import plot
import random
import csv
import sys

class  Sample:
    def __init__(self, features, low_features, label):
        self.features = np.array(features)
        self.low_features = np.array(low_features)

class RandomProjection:

    def __init__(self, k, d):
        # generate matrix R k x d
        coefficient = 1/math.sqrt(d)
        self.R = [[ self.pom()*coefficient for x in range(d)] for x in range(k)]

    def project( self, obj ):
        return np.dot( self.R, obj );

    # Plus or minus
    def pom(self):
        if random.random() < 0.5:
            return -1
        return 1

# End declaration

def run():
    parser = OptionParser()
    parser.add_option("-k", "--dimension", dest="k", type="int", default=50 )
    parser.add_option("-n", "--data", dest="n", type="int", default=20 )

    (opt, args) = parser.parse_args()

    (train, validation, test) = data_loader.load_data_wrapper();

    train = train[:opt.n]

    dim = len(train[0][0])

    print( "k: %d" % ( opt.k ) )
    print( "dimension: %d" % (dim))
    print("---")

    R = RandomProjection( opt.k, dim )

    train = [ Sample( t[0], R.project(t[0]), t[1] ) for t in train ]

    total = 0;
    f = open( "k-"+str(opt.k)+".csv", 'wt')
    try:
        writer = csv.writer(f)
        writer.writerow( ('Instance 1', 'Instance 2', 'Distortion') )
        for i in range(len(train)):
            a = train[i]
            for j in range(i+1,len(train)):
                if i == j:
                    continue

                total+=1
                b = train[j]
                dist_k_dim = np.linalg.norm( a.low_features - b.low_features )
                dist_d_dim = np.linalg.norm( a.features - b.features )
                distortion = dist_k_dim / dist_d_dim
                writer.writerow( (i+1, j+1, "%.4f" % ( distortion ) ))
    finally:
        f.close()
    # total should be C(20,2)
    print(total)

if __name__ == "__main__":
   run()

