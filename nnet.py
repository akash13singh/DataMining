from openml.apiconnector import APIConnector
import os
from sklearn import cross_validation as cv
from matplotlib import pyplot as plt
from time import time
from sklearn.metrics import classification_report,zero_one_loss
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import numpy as np
from sklearn.neural_network import MLPClassifier
from datetime import datetime
from sklearn import preprocessing
from operator import itemgetter

def load_data(dataset_id):
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
    print("no. of samples :"+str(len(X)))
    return (X,y,attribute_names)

def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def run():
    X, y, attribute_names = load_data(554)
    X_train, X_test = X[:60000], X[60000:]
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)
    y_train, y_test = y[:60000], y[60000:]
    print(len(X_train))
    print(len(y_test))
    print(attribute_names)

    '''mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10000,batch_size=600,
        algorithm='sgd',  random_state=1, alpha=.1,tol= .0001,
        learning_rate_init=.01,activation="logistic",learning_rate="constant",momentum=.9)
    '''

    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10000,batch_size=600,
                        algorithm='sgd',  random_state=1,activation="logistic",learning_rate="constant")
    mlp.fit(X_train, y_train)
    print("Training set score: %f" % mlp.score(X_train, y_train))
    print("Test set score: %f" % mlp.score(X_test, y_test))
    param_grid = { "alpha":[.1,.01,.001],
                   "momentum":[.7,.9],
                   "learning_rate_init":[.1,.01,.001]
    }
    

    mlp= MLPClassifier(hidden_layer_sizes=(50,), max_iter=10000,batch_size=600,
                        algorithm='sgd',  random_state=1,activation="logistic",learning_rate="constant")
    random_search = RandomizedSearchCV(mlp, param_distributions= param_grid,n_iter = 8)
    start = time()
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start),8))
    report(random_search.grid_scores_,10)
    best = random_search.best_estimator_

    print("Test set score of best nnet: %f" % best.score(X_test, y_test))

if __name__ == "__main__":
    run()