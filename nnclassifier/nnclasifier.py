import mnist_dataloader as data_loader
import time
import numpy
import pandas
import random
import pickle
import math
from optparse import OptionParser

from scipy import linalg

NUM_LABEL = 10
SEED = 71


class RandomProjection:

    def __init__(self, k, d):
        print("k: "+str(k))
        print("d: "+str(d))
        # generate matrix R k x d
        coefficient = 1/math.sqrt(d)
        self.R = [[ self.pom()*coefficient for x in range(d)] for x in range(k)]

    def project( self, obj ):
        return numpy.dot( self.R, obj );
        numpy.dot

    # Plus or minus
    def pom(self):
        if random.random() < 0.5:
            return -1
        return 1

#class for training/test sample object
class  Sample:
    def __init__(self,features,label):


        # do not normalise feature vector because of euclidean distance
        #self.features = features.flatten() / linalg.norm(features)

        self.features = features.flatten()

        self.label=-1

        #set label
        if(isinstance( label, numpy.int64 )):
            self.label = label
        else:
            for k in range(10):
                if( label[k] == 1):
                    self.label = k


def runNNClassifier(train,test,dist_type, reduce_dim_flag, new_dims):

    if reduce_dim_flag[0] == "y":
         train = list(train)
         print("size of train "+str(len(train)))
         R = RandomProjection( new_dims, len(train[0][0]))
         train = [ Sample(  R.project(t[0]), t[1] ) for t in train ]
         test =  [ Sample(  R.project(t[0]), t[1] ) for t in test ]
         print(len(train))
         print(len(test))
    else :
    # Convert data to "Sample" object
        train = [ Sample( t[0], t[1] ) for t in train ]
        test = [ Sample( t[0], t[1] ) for t in test ]

    # Set seed for shuffling
    # This guarantees that the shuffled set will be the same everytime.
    random.seed(SEED)

    # Randomly select a subset of testing set
    # as classing the whole set takes too long to compute.
    #random.shuffle(test)
    # test = test[0:10]

    print("Training set size: %d" % len(train))
    print("Testing set size: %d" % len(test))

    test_counter =0;
    start_time = time.time()

    # Find nearest neighbor for all testing sample
    for i in test:
        test_counter+=1
        if test_counter % 5000 == 0 :
            print(">> " + str(test_counter))

        # Set the first sample of training set as the nearest one.
        nearest_sample = train[0]

        min_distance = calculate_distance(dist_type,i,nearest_sample)

        # Iterate through all training set to find the nearest neighbor
        for j in train:
            #distance = cosine_distance_numpy(i,j)

            distance = calculate_distance(dist_type,i,j)
            if distance < min_distance:
                min_distance = distance
                nearest_sample = j;

            # Stop further process if the min_distance is zero
            if min_distance == 0:
                break

        #set predicted label to the label of nearest neighbor
        i.predicted_label = nearest_sample.label

    end_time = time.time()

    # Compute confusion_matrix, accuracy and prediction and recall for each label
    print("----- Confusion Matrix -----")
    matrix = confusion_matrix( test )
    print("%s" % ( pandas.DataFrame( matrix ) ))
    print("----------------------------")
    print("Accuracy : %0.2f" % ( accuracy(matrix) ))

    for i in range(NUM_LABEL):
        print("Label %d : precision: %.2f \t recall: %.2f" % ( i, precision( matrix, i ), recall( matrix, i ) ))

    print ("----------------")
    time_diff = end_time - start_time
    print("Time spent : %.2fs ( %.2fs per sample )" % ( time_diff, time_diff*1.0/len(test) ))

    # Save the result for further analysis
    output = open('predicted.pkl', 'wb')
    pickle.dump( test, output )
    output.close()

def cosine_distance(v1,v2):
    dot_product = 0
    for i in range(len(v1.features)):
        dot_product += ( v1.features[i]*v2.features[i] )
    similarity = dot_product / ( 1.0*( v1.length * v2.length ) )

    return 1-similarity

# To compute dot product efficiently, we use numpy package to compute
# dot product. Since feature vector was normalized already when creating the Sample object,
# we can use the dot product directly to compute the cosine distance
def cosine_distance_numpy(v1,v2):
    #print(numpy.shape(v1.features))
    #print(numpy.shape(v2.features))
    #return 1 - numpy.dot(v1.features/linalg.norm(v1.features),v2.features/linalg.norm(v2.features))
    return 1 - numpy.dot(v1.features,v2.features)


#function to compute dot educlidean distance
def euclidean_distance(v1,v2):
    dist = numpy.linalg.norm(v1.features-v2.features)
    return dist

def calculate_distance(dist_type,v1,v2):
    if dist_type =="euclidean":
        return euclidean_distance(v1,v2)
    if dist_type =="cosine":
        return cosine_distance_numpy(v1,v2)
    return "error"

def confusion_matrix(data):
    # Initialize empty confusion matrix
    # Rows are actual labels and columns are predicted labels
    matrix = []
    for r in range(NUM_LABEL):
        matrix.append([0]*NUM_LABEL)

    # Compute the matrix
    for i in data:
        temp = matrix[i.label][i.predicted_label]
        matrix[i.label][i.predicted_label] = temp + 1

    return matrix

def matrix(data):
    # Initialize empty confusion matrix
    # Rows are actual labels and columns are predicted labels
    matrix = []
    for r in range(NUM_LABEL):
        matrix.append([0]*NUM_LABEL)


    # Compute the matrix
    for i in data:
        temp = matrix[i.label][i.predicted_label]
        matrix[i.label][i.predicted_label] = temp + 1

    header = str("     |" + "%5d |"*10 + " Recall") % tuple(range(10))
    print(header)
    print("-"*len(header))

    precision = [0]*10
    for i in range(10):
        correct_classify = matrix[i][i]*1.0
        recall =  0 if correct_classify == 0 else correct_classify / sum( matrix[i] )
        print(str(" %3d |"+"%5d |"*10+"  %.2f") % tuple([i] + matrix[i] +[recall] ))

        label_precision = 0
        for j in range(10):
            precision[i] = precision[i] + matrix[j][i]
        precision[i] = correct_classify / precision[i]

    print("-"*len(header))
    print(str("Prec |" + " %.2f |"*10) % tuple( precision ))

def precision( confusion_matrix, class_name ):
    bucket = []

    # Sum along a column to compute precision of a particular class
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
    print(text)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d","--dist_metric: 1.cosine   2.euclidean", dest="dist_type", type="string" ,default ="euclidean")
    parser.add_option("-r","--reduce dimensions: Y(es) ,N(o)", dest="reduce_dims_flag", type="string", default="yes")
    parser.add_option("-k", "--new_dimensions", dest="k", type="int", default=100 )
    parser.add_option("-n", "--train_data_len", dest="n", type="int", default=1000)
    parser.add_option("-t", "--test_data_len", dest="t", type="int", default=100)

    (opt, args) = parser.parse_args()
    print("distance_metric: "+opt.dist_type)
    print("reduce_dims_flag: "+opt.reduce_dims_flag)
    print("data lenght: "+str(opt.n))
    print("new dinmesnions: "+str(opt.k))


    (train, validation, test) = data_loader.load_data_wrapper(opt.n,opt.t)
    runNNClassifier(train,test,opt.dist_type, opt.reduce_dims_flag.lower(), opt.k)
