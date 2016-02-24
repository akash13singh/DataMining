import random
import numpy as np
def find_centers( k, data ):
    data = np.copy(data)
    first_point = random.randint(0, len(data) )
    ( data, element ) = pop_from_numpy( data, first_point )
    centers = [ element ]

    for d in range(k-1):
        # print d
        length = len(data)
        dist_arr = np.array([0]*length)
        for p in range(len(data)):
            point = data[p]
            min_dist = np.linalg.norm( centers[0].features- point.features )
            for c in centers[1:]:
                dist = np.linalg.norm( point.features - c.features )
                if dist < min_dist:
                    min_dist = dist
            dist_arr[p] = (1.0*min_dist)**2

        sum_dist  = sum(dist_arr)*1.0
        alpha_arr = dist_arr/sum_dist

        cumulative_alpha = [0]*(length+1)

        for i in range(length):
            cumulative_alpha[i+1] = cumulative_alpha[i] + alpha_arr[i]

        r = random.uniform(0,1);
        # print cumulative_alpha
        # print r
        for i in range(1,len(cumulative_alpha)):
            if( cumulative_alpha[i-1] <=  r and r < cumulative_alpha[i] ):
                ( data, element ) = pop_from_numpy( data, i-1 )
                # print '--------'
                centers.append( element )
                break
    return centers

def pop_from_numpy( arr, index ):
    element = arr[index];
    new_arr = np.delete( arr, index )
    return ( new_arr, element )
