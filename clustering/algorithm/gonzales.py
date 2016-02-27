import math

def find_centers( k, data ):
    n = len(data)
    print(n)
    print(data[n-1].features)

    clusters = [0]

    #iterate for the required number of clusters
    for num_center in range(1,k):
        max_cost = 0
        new_center = 0
        #iterate over data
        for i in range(1,n):
            # if i in clusters:
            #     continue
            #find minimum distance of point i from cluster centre
            min_distance = calculate_distance(data[i],data[clusters[0]])
            for j in range(1,num_center):
                distance =  calculate_distance(data[i],data[clusters[j]])
                if distance < min_distance:
                    min_distance = distance
            if i == 1003:
                print min_distance,i

            if min_distance > max_cost:
                max_cost = min_distance
                new_center = i
        clusters.append(new_center)

    for m in range(k):
        clusters[m] = data[clusters[m]]

    return clusters


def calculate_distance(p1,p2):
    dist = 0
    dims = len(p1.features)
    for i in range(dims):
        dist += math.pow(p1.features[i]-p2.features[i],2)
    dist = math.sqrt(dist)
    return dist
