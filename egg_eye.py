from sklearn import ensemble
#from openml.apiconnector import APIConnector
from openml import tasks,runs
import xmltodict
import csv
import os
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import zero_one_loss
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.neural_network import MLPClassifier


def load_data(dataset_id):
    csv = np.genfromtxt ('/home/akash/eye-egg.csv', delimiter=",")
    X = csv[:,:-1]
    y= csv[:,-1]
    print(X[0])
    print(y[1])
    return X,y
    '''
    #openml connection
    home_dir = os.path.expanduser("~")
    openml_dir = os.path.join(home_dir, "openml")
    cache_dir = os.path.join(openml_dir, "cache")
    print(openml_dir)
    with open(os.path.join(openml_dir, "apikey.txt"), 'r') as fh:
        key = fh.readline().rstrip('\n')
    openml = APIConnector(cache_directory=cache_dir, apikey=key)
    dataset = openml.download_dataset(dataset_id)
    # load data into panda dataframe
    X, y, attribute_names = dataset.get_dataset(target=dataset.default_target_attribute, return_attribute_names=True)
    X,y = shuffle(X,y)
    print("no. of samples :"+str(len(X)))
    return (X,y,attribute_names)
    '''

def run():
    X,y = load_data(1471)
    y=y-1
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3,random_state=345)
    #clf = AdaBoostClassifier(   n_estimators=1000)
    clf = RandomForestClassifier(warm_start=True,n_estimators=128,criterion="entropy",min_samples_split=20,bootstrap=True,random_state=123  )
    #clf = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000, batch_size=1000,
    #                 algorithm='sgd', random_state=1, activation="logistic", learning_rate="constant")
    #clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME",n_estimators=200)

    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    print("Testing Error %s"%(zero_one_loss(y_test,y_predicted)))
    print("Testing accuracy %s"%(clf.score(X_test,y_test)))
    scores_auc = cross_validation.cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
    scores_accuracy = cross_validation.cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("CV AUC %f"%(scores_auc.mean()))
    print("CV accuracy %f" % (scores_accuracy.mean()))

if __name__ == "__main__":
    run()
