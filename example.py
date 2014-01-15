#!/usr/bin/python

import os
import numpy as np
from sklearn.decomposition import PCA
import time
from sklearn.cluster import k_means
from elm import ELMClassifier, ELMRegressor, SimpleELMClassifier, SimpleELMRegressor
from random_hidden_layer import SimpleRandomHiddenLayer, RBFRandomHiddenLayer
from sklearn import cross_validation
from sklearn import tree
from sklearn.metrics import confusion_matrix
import random

def get_uci_path():
    """
    Returns the path to the UCI datasets, depending on the 
    host name.

    :return: string
    """
    import socket
    hn = socket.gethostname()

    if hn == 'corsair':
        path = '/home/alexandre/Dropbox/Documents/work/borjaayerdi/pythoncode/uci-datasets'

    elif hn == 'Ayerdi-PC':
        path = 'uci-datasets'

    return path


def read_uci_dataset(base_dir, dataset_idx=1):
    """
    This function returns the path to the UCI dataset

    :param base_dir: string
    The path to where the UCI datasets folder are.

    :param dataset_idx: int
    The number of the dataset

    :return: 2-tuple of ndarray
    The dataset in a Numpy array,
    the first is the data samples and the second, the labels
    """

    #full_path = os.path.join(base_dir, 'uci-datasets')

    # Load the data
    path = os.path.join(base_dir, str(dataset_idx) + '.data')

    data = np.genfromtxt(path, delimiter=",")
    rows, cols = data.shape

    # Delete the first column of labels
    Y = data[:, 0]
    X = data[:, 1:rows]

    return X, Y

def apply_pca(data, labels, n_comps=1):
    """
    Applies PCA to the data

    :param data: ndarray
    A MxN array with M samples of dimension N

    :param labels: ndarray or list
    A 1xN array with the class labels of each sample
    of data

    :return: sklearn.decomposition.PCA

    """
    # PCA
    pca = PCA(n_components=n_comps, whiten=False, copy=True)
    pca.fit(data)

    return pca
    
def clasProbDist(data,labels,dim):
    """
    Will return a vector with the composition, type of classifiers, of the ensemble.
    
    """
    
    rankscore =[55,34,21,13,8,5,3,2,1,1] # Fibbonacci
    
    # Train test split
    x_train, x_test, \
    y_train, y_test = cross_validation.train_test_split(data, labels, test_size=0.2)
    
    # Matrix to order
    matrix = []
    
    # Train/Test DT == 0
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    matrix.append([clf.score(x_test, y_test),0])
    
    #  Train/Test ELM == 1
    hiddenN = 1000
    if len(y_train)/3 < hiddenN:
        hiddenN = len(y_train)/3
        
    elmc = SimpleELMClassifier(n_hidden=hiddenN)
    elmc.fit(x_train, y_train)
    matrix.append([elmc.score(x_test, y_test),1])
    
    # Sort with the error of the classification
    matrix.sort()
    matrix.reverse()
    
    probDist = []
    for i in range(0,len(matrix)):
        a = matrix[i]
        a = a[1]
        probDist.extend(np.tile(a, rankscore[i]))
        
    ensembleComposition = []
    for j in range(0,dim):
        randInd = random.randint(0, len(probDist)-1)
        selectedClass = probDist[randInd]
        ensembleComposition.append(selectedClass)

    return ensembleComposition
    
    
    
    
# dim = 35 ponemos para pruebas 2
def trainADAHERF(X, Y, dim=2):
    """
    Train AdaHERF algorithm
    """
    n_samps, NF = X.shape

    # Train test split
    x_train, x_test, y_train,y_test = cross_validation.train_test_split(X, Y, test_size=0.2)
    
    # We save x_test and y_test for testing => 20% of the total data.
    # From the 80% of training data we use 30% for ensemble model selection and 70% for real training.
    x_train, x_trainADAHERF, \
    y_train, y_trainADAHERF = cross_validation.train_test_split(x_train, y_train, test_size=0.7)
    
    # We generate ensemble composition
    ensembleComposition = clasProbDist(x_train, y_train, dim)
    
    # We will use x_trainADAHERF and y_trainADAHERF for adaHERF training
    # And x_test & y_test for testing
    
    for i in range(0,dim):
        # For each classifier in the ensemble
        # Given:
        # X: the objects in the training data set (an N x n matrix)
        # Y: the labels of the training set (an N x 1 matrix)
        # K: the number of subsets
        # NF: the number of total features
        # {w1,w2,.., wc}: the set of class labels
        #L

        # Prepare the rotaton matrix R:
        # Split F (the feature set) into K subsets Fij (for j=1,..,K/4)
        # K is a random value between 1 and NF/4.
        # We want at least 1 feature for each subset.
        K = int(round(1 + NF/4*random.random()))
        
        FK = np.zeros((K,NF))
        for j in range(0,K):
            numSelecFeatures = int(1 + round((NF-1)*random.random()))
            rp = np.random.permutation(NF)
            
            v = [rp[k] for k in range(0, numSelecFeatures)]
            #v = []
            #for k in range(0,numSelecFeatures):
            #    v.append(rp[k])
            
            FK[j,v] = 1
    
        
        R = np.zeros((NF,NF))
        for l in range(0,K):
            # Let Xij be the data set X for the features in Fij
            pos = np.nonzero(FK[l,:])[0]

            vpos = [pos[m] for m in range(0, len(pos))]
            #vpos = []
            #for m in range(0,len(pos)):
            #    vpos.append(pos[m])
            
            XXij = X[:, vpos]
            pca = apply_pca(XXij, Y, len(pos))
    
            # pca.components_ is the rotation matrix.
            #rotate = XXij.dot(pca.components_)
            
    
    
    
    # herf es un objeto con todos los clasificadores clf y elmc.
    # inforotar tiene dentro todas las matrices de rotacion
    # ensembleComposition es un array con los clasificadores del ensemble por tipo.
    herf = []
    inforotar = []
    
    return herf, inforotar, ensembleComposition
    
    
    
    
    
#Desde arriba, hasta aqui, lo copias y lo pegas al ipy, a parte.

#esto se pone para cuando este fichero se
#va a ejecutar como un script, desde fuera
#Lo que hay aqui debajo lo puedes poner en otro sitio.
if __name__ == '__main__':

    uci_path = get_uci_path()
    X, Y = read_uci_dataset(uci_path)
    herf, inforotar, ensembleComposition = trainADAHERF(X,Y)

    
    x_train = X
    y_train = Y
    x_test = X
    y_test = Y
    
    # Decision tree
    start = time.time()
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    # We get the predicted classes
    #clf.predict(x_test)
    end = time.time()
    print "DT accuracy"
    print clf.score(x_test, y_test)
    print "DT time"
    print end - start
    
    # We are going to try ELM
    start = time.time()
    hiddenN = 1000
    if len(y_train)/3 < hiddenN:
        hiddenN = len(y_train)/3
        
    elmc = SimpleELMClassifier(n_hidden=hiddenN)
    elmc.fit(x_train, y_train)
    end = time.time()
    print "ELM accuracy"
    # We get the predicted classes
    #elmc.predict(x_test)
    print elmc.score(x_test, y_test)
    print "ELM time"
    print end - start
	

    
