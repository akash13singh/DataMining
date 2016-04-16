import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import cross_validation
import numpy as np
import os

def load_data():
    csv = np.genfromtxt('/home/akash/spam.csv', delimiter=",")
    X = csv[:, :-1]
    y = csv[:, -1]
    print(len(y))
    return X, y

def run():
    errors= []
    forest_size =[]
    num_forests = 11
    X, y = load_data()
    random_forest = RandomForestClassifier(warm_start=True,bootstrap=True,oob_score=True,random_state=123)
    for i in range(num_forests):
        print(pow(2,i))
        size = pow(2,i)
        random_forest.set_params(n_estimators=size)
        random_forest.fit(X,y)
        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - random_forest.oob_score_
        print(oob_error)
        errors.append(oob_error)
        forest_size.append(size)



    plt.plot(range(num_forests),errors)
    #plt.xlim(forest_size[0], forest_size[num_forests-1])
    plt.xticks(range(num_forests),forest_size,rotation=45)
    plt.xlabel("num_trees (2^x) ")
    plt.ylabel("OOB error rate")
    plt.show()

    '''
    cv_errors = []
    forest_size = []
    random_forest = RandomForestClassifier(warm_start=True, random_state=123)
    for i in range(num_forests):
        print(pow(2, i))
        size = pow(2, i)
        random_forest.set_params(n_estimators=size)
        scores = cross_validation.cross_val_score(random_forest, X, y,cv = 10, scoring = 'accuracy')
        cv_error = 1 - scores.mean()
        print(cv_error)
        cv_errors.append(cv_error)
        forest_size.append(size)
    plt.plot(range(num_forests), cv_errors)
    # plt.xlim(forest_size[0], forest_size[num_forests-1])
    plt.xticks(range(num_forests), forest_size, rotation=45)
    plt.xlabel("num_trees (2^x) ")
    plt.ylabel("cv error rate")
    plt.show()
    '''

if __name__ == "__main__":
    run()
