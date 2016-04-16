from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc,accuracy_score

def load_data():
    csv = np.genfromtxt('/home/akash/spam.csv', delimiter=",")
    X = csv[:, :-1]
    y = csv[:, -1]
    return csv

def run():
    result={}
    p1=0
    p0=0
    data= load_data()
    arr = np.array([[1,42,31,64,1],[2,98,76,12,0],[3,90,72,53,1]])
    #size of dataset
    n= len(data)
    # no of bootsptrap samples
    N = 100
    num_forests= 10
    i = 1
    row_num = np.arange(0,n,1)
    p1_arr  = np.zeros(n)
    oob_count_arr = np.zeros(n)
    miss_arr = np.zeros(n)
    data = np.insert(data, 0,  row_num, axis=1)
    variances = []
    aucs= []
    biases = []
    forest_size = []
    errors = []
    np.random.seed(341)
    for i in range(1,num_forests+1):

        size = pow(2, i)
        random_forest = RandomForestClassifier(bootstrap=True,n_estimators=size,warm_start=False)
        cumulative_set_diff = 0
        sum_auc = 0
        sum_error = 0
        for j in range(N):
            bootstrap_indexes = np.random.choice(n,n,replace=True)
            oob_indexes= np.setdiff1d(row_num,bootstrap_indexes)
            training = data[bootstrap_indexes]
            test = data[oob_indexes]
            #print(bootstrap_indexes)
            #print(oob_indexes)
            #print(training[:,0])
            #print(test[:,0])

            random_forest.fit(training[:,1:-1],training[:,-1])
            predicted_test = random_forest.predict(test[:,1:-1])
            predicted_prob = random_forest.predict_proba(test[:,1:-1])
            error = 1 - accuracy_score(test[:,-1],predicted_test)
            sum_error = sum_error+error
            fpr,tpr,thresholds = roc_curve(test[:, -1], predicted_prob[:,1])
            roc_auc = auc(fpr, tpr)
            sum_auc = sum_auc+ roc_auc
            #print(predicted_test)
            #print(test[:,-1])
            #exit(0)
            #print(accuracy_score(predicted_test,test[:,-1]))

            for index in range(len(test)):
                elem = test[index][0]
                if predicted_test[index] != test[index, -1]:
                    miss_arr[elem] = miss_arr[elem]+1
                if predicted_test[index]==1:
                    p1_arr[elem] = p1_arr[elem]+1
                oob_count_arr[elem] = oob_count_arr[elem]+1

            set_diff_ratio = len(oob_indexes)/n
            #print(set_diff_ratio)
            cumulative_set_diff = cumulative_set_diff + set_diff_ratio

        #print("average bootsrap ratio %f" % (cumulative_set_diff / N))
        sample_bias = 0
        sample_variance=0
        for k in range(n):
            oob_count = oob_count_arr[k]
            p1= p1_arr[k]/oob_count
            p0 = 1  - p1
            miss_rate = miss_arr[k]/oob_count
            weight = oob_count/N
            sample_bias = sample_bias +  pow(miss_rate,2)
            sample_variance = sample_variance + (p0*p1)

        squared_bias = sample_bias/n
        variance = sample_variance/n
        biases.append(squared_bias)
        forest_size.append(size)
        variances.append(variance)
        aucs.append(sum_auc/N)
        errors.append(sum_error/N)
    plt.subplot(2,1,1)
    plt.plot(range(1,num_forests+1),variances,"r-",label="variance")
    plt.plot(range(1, num_forests + 1), biases,"b-",label="bias")
    plt.plot(range(1, num_forests + 1), errors, "g-", label="error")
    plt.xticks(range(1,num_forests+1))
    plt.xlabel("num_trees (2^x) ")
    plt.ylabel("bias and variance")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.subplot(2,1,2)
    plt.plot(range(1, num_forests + 1), aucs, "r-", label="auc")
    plt.xticks(range(1, num_forests + 1))
    plt.xlabel("num_trees (2^x) ")
    plt.ylabel("auc")
    plt.show()

if __name__ == "__main__":
    run()