import math

def find_centers( k, data ):
    n = len(data)
    print(n)
    clusters = []

    #set first cluster centre to 0
    clusters.append(0)

    #iterate for the required number of clusters
    for num_center in range(1,k):
        max_cost = 0
        new_center = 0
        #iterate over data
        for i in range(1,n):
            '''
            if i in clusters:
                continue
            '''
            #find minimum distance of point i from cluster centre
            min_distance = calculate_distance(data[i],data[clusters[0]])
            for j in range(0,num_center):
                distance =  calculate_distance(data[i],data[clusters[j]])
                if distance < min_distance:
                    min_distance = distance

            #find min_distance is greater than max_cost update max cost and center
            if min_distance > max_cost:
                max_cost = min_distance
                new_center = i
        clusters.append(new_center)


    for m in range(k):
        clusters[m] = data[m]

    print(clusters)
    return clusters


def calculate_distance(p1,p2):
    dist = 0
    dims = len(p1.features)
    for i in range(dims):
        dist+= math.pow(p1.features[i]-p2.features[i],2)
    dist = math.sqrt(dist)
    return dist
