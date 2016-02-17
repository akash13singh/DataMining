import mnist_dataloader as data_loader
import time
import numpy
import pandas
import random

from scipy import linalg

NUM_LABEL = 10
SEED = 71

class  Sample:
    def __init__(self,features,label):

        # normalise feature vector
        self.features = features.flatten() / linalg.norm(features)

        self.label=-1

        if(isinstance( label, numpy.int64 )):
            self.label = label
        else:
            for k in range(10):
                if( label[k] == 1):
                    self.label = k


def runNNClassifier():
    (train, validation, test) = data_loader.load_data_wrapper();

    # Convert data to "Sample" object
    train = [ Sample( t[0], t[1] ) for t in train ]
    test = [ Sample( t[0], t[1] ) for t in test ]

    # This guarantees that the shuffled set will be the same everytime.
    random.seed(SEED)

    # Randomly select a subset of testing set
    # as classing the whole set takes too long to compute.
    random.shuffle(test)
    test = test[0:1000]

    print "Training set size: %d" % len(train)
    print "Testing set size: %d" % len(test)

    test_counter =0;
    start_time = time.time()

    for i in test:
        test_counter+=1
        if test_counter % 10 == 0 :
            print test_counter
        nearest_sample = train[0]
        min_distance = cosine_distance_numpy(i,nearest_sample)

        for j in train:
            distance = cosine_distance_numpy(i,j)
            if distance < min_distance:
                min_distance = distance
                nearest_sample = j;
            if min_distance == 0:
                break

        i.predicted_label = nearest_sample.label

    end_time = time.time()

    # Compute confusion_matrix, accuracy and prediction and recall for each label
    print "----- Confusion Matrix -----"
    matrix = confusion_matrix( test )
    print "%s" % ( pandas.DataFrame( matrix ) )
    print "----------------------------"
    print "Accuracy : %0.2f" % ( accuracy(matrix) )

    for i in range(NUM_LABEL):
        print "Label %d : precision: %.2f \t recall: %.2f" % ( i, precision( matrix, i ), recall( matrix, i ) )

    print "----------------"
    time_diff = end_time - start_time
    print "Time spent : %.2fs ( %.2fs per sample )" % ( time_diff, time_diff*1.0/len(test) )

def cosine_distance(v1,v2):
    dot_product = 0
    for i in range(len(v1.features)):
        dot_product += ( v1.features[i]*v2.features[i] )
    similarity = dot_product / ( 1.0*( v1.length * v2.length ) )

    return 1-similarity

def cosine_distance_numpy(v1,v2):
    return 1 - numpy.dot(v1.features,v2.features)


def confusion_matrix(data):
    # Initialize empty confusion matrix
    matrix = []
    for r in range(NUM_LABEL):
        matrix.append([0]*NUM_LABEL)

    for i in data:
        temp = matrix[i.label][i.predicted_label]
        matrix[i.label][i.predicted_label] = temp + 1

    return matrix

def precision( confusion_matrix, class_name ):
    bucket = []
    for i in range(NUM_LABEL):
        bucket.append( confusion_matrix[i][class_name] )

    all_predict = sum(bucket)
    correct_predict = confusion_matrix[class_name][class_name]

    precision = 0
    if( all_predict != 0 ):
        precision = correct_predict*1.0 / all_predict

    return precision;

def recall( confusion_matrix, class_name ):
    bucket = confusion_matrix[class_name]

    instance_in_class = sum(bucket)
    correct_predict = confusion_matrix[class_name][class_name]

    recall = 0
    if( instance_in_class != 0 ) :
        recall = 1.0*correct_predict/instance_in_class

    return recall

def accuracy( confusion_matrix ):
    total_instance = 0
    correct_predict = 0
    for i in range(NUM_LABEL):
        total_instance  = total_instance + sum( confusion_matrix[i] )
        correct_predict = correct_predict + confusion_matrix[i][i]
    return correct_predict*1.0 / total_instance

def log( format, data=() ):
    text = format % data
    print text

if __name__ == '__main__':
    runNNClassifier()
