import numpy as np
import math
import matplotlib.pyplot as plt

data =[]


#clusters = []

def load_data_1b(fpath):
    f = open(fpath, 'r')
    for line in f:
        words = line.split()
        data.append(words)
    f.close()
    arr = np.array(data, dtype=np.float64)
    return arr[:, 1:]


def calculate_kmedian_cost(p1,p2):
    cost = 0
    global total_cost
    for i in range(len(p1)):
        cost += math.pow(float(p1[i])-float(p2[i]),2)
    cost = math.sqrt(cost)
    return cost

def calculate_new_medians(clusters):
    new_medians=[]
    for x in sorted(np.unique(data[...,6])):
        new_medians.append([np.median(data[np.where(data[...,6]==x)][...,1].astype(np.float)),
        np.median(data[np.where(data[...,6]==x)][...,2].astype(np.float)),
        np.median(data[np.where(data[...,6]==x)][...,3].astype(np.float)),
        np.median(data[np.where(data[...,6]==x)][...,4].astype(np.float)),
        np.median(data[np.where(data[...,6]==x)][...,5].astype(np.float))])
    return new_medians

def create_clusters(medians,num_clusters):
    global data
    clusters = []
    cluster_arr=[]
    for i in range(num_clusters):
        clusters.append([])
    total_cost = 0
    for i in range(len(data)):
        #find nearest median and assign cluster
        min_cost = calculate_kmedian_cost(data[i][1:5],medians[0])
        cluster = 0
        count =1
        for j in medians[1:]:
            cost = calculate_kmedian_cost(data[i][1:5],j)
            if cost < min_cost:
                min_cost = cost
                cluster = count
            count+=1
        clusters[cluster].append(i)
        cluster_arr.append(cluster)
        total_cost += min_cost

    data=np.insert(data,6,cluster_arr,1)

    return total_cost,clusters


def run():
    num_clusters = 4
    median_indices = []
    data = load_data_1b("./data1b/C3.txt")

    #initialize median index
    np.random.seed(37)
    median_indices = np.random.randint(0,len(data),num_clusters)
    median_indices = median_indices.tolist()
    medians = []
    print("Initial Median Indices"+str(median_indices))
    for i in range(num_clusters):
        medians.append(data[median_indices[i]])

    cost_k_median,clusters = create_clusters(medians,num_clusters)

    z=1
    while(True):
        print("---------------Iteration "+str(z)+"------------------------")
        print("medians: "+str(medians))
        print("clusters: "+str(clusters))
        print("cost :: %f",cost_k_median)

        new_medians = calculate_new_medians(clusters)

        print("new_medians "+str(new_medians))
        new_cost_k_median,new_clusters = create_clusters(new_medians,num_clusters)
        print("new cost %f",new_cost_k_median)

        if new_cost_k_median >= cost_k_median:
            break
        median = new_medians
        clusters = new_clusters
        cost_k_median = new_cost_k_median
        z=z+1



if __name__ == "__main__":
   run()

#(0.09771348879-0.82443692)^2+	(1.856078237- 7.78300883)^2 +	(2.100760575-0.29411489)^2	+ (9.251076729+0.92552974)^2	+(-1.327820728-0.36975157)^2