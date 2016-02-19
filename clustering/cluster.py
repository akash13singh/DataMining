#!/usr/local/bin/python
from algorithm import naive
from algorithm import gonzales
from optparse import OptionParser

import dataloader_1b as dt
import math
import numpy as np

class  Sample:
    def __init__(self,features):
        self.features = features

def print_samples( set_data ):
    print "["
    for d in set_data:
        arr = []
        if d is not None:
            arr = d.features
        print "\t %s" %( arr )
    print "]"

def mean_cost( p1, p2 ):
    sos = 0
    dim = len(p1.features)
    for d in range(dim):
        sos = sos + math.pow(p1.features[d] - p2.features[d],2)
    return math.sqrt(sos)

def empty_clusters(k):
    clusters = []
    for i in range(k):
        clusters.append([])
    return clusters

def mean_center(cluster):
    if len(cluster) == 0 :
        return None

    dim            = len(cluster[0].features)
    sum_dimensions = [0]*dim
    for p in cluster:
        for i in range(dim):
            sum_dimensions[i] = sum_dimensions[i] + p.features[i]

    for i in range(dim):
        sum_dimensions[i] = sum_dimensions[i]*1.0 / len(cluster);

    # New center
    return Sample( sum_dimensions )

def is_same_point_set( s1, s2):
    dim = 2
    for i in range(len(s1)):
        same_value = True
        for d in range(dim):
            same_value = same_value and np.isclose(s1[i].features[d], s2[i].features[d], 0.0001 )
        if( not same_value ):
            return False
    return True

# End declaration

parser = OptionParser()
parser.add_option("-a", "--algorithm", dest="algorithm",
    help="Algorithm for initialization center points", default="naive"
)
parser.add_option("-k", "--cluster", dest="k", type="int", default=2 )

(opt, args) = parser.parse_args()

print "k: %d" % ( opt.k )
print "algorithm: %s" % ( opt.algorithm )
print "loss-function: %s" % ( "mean" )
print "---"

# Load data here

data = dt.load_data_1b("./data1b/C2.txt")
data = [ Sample( d ) for d in data ]

# Find initial centers
centers = eval(opt.algorithm).find_centers( opt.k, data );
total_cost = 0

clusters = empty_clusters(opt.k)

iteration = 1
while True:
    total_cost = 0
    # Assign data to clusters
    for d in data:
        d.cluster = 0
        cost  = mean_cost( d, centers[d.cluster] )
        for c in range(len(centers[1:])):
            new_cost = mean_cost( d, centers[c] )
            if new_cost < cost :
                cost      = new_cost
                d.cluster = c
            elif new_cost == 0:
                d.cluster = c
                cost      = 0
                break

        clusters[d.cluster].append(d)
        total_cost = total_cost + cost

    # Find new center
    new_centers = [0]*opt.k
    for c in range(opt.k):
        new_centers[c] = mean_center( clusters[c] )
        if( new_centers[c] is None ):
            new_centers[c] = centers[c]

    # Stop if loss doesn't change anymore.
    # print_samples( centers )
    # print_samples( new_centers )
    if is_same_point_set( new_centers, centers ) :
        break
    else:
        centers = new_centers

        # Print debug msg
        s = "iteration " + str( iteration ) + " : "
        for c in centers:
            s = s + str(c.features)
        print s

    iteration = iteration + 1

print "---"
index = 0
for d in data:
    print str(index)+":"+ str(d.features) + ":"+str(d.cluster)
    index = index + 1

print "---"
print "total_cost: %4f" % ( total_cost )
print "iteration: %d" % ( iteration )
