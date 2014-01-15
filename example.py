#!/usr/bin/python

import os
import time
import random
import numpy as np

from scipy.stats import mode

from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.cluster import k_means
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix

from elm import ELMClassifier, ELMRegressor, SimpleELMClassifier, SimpleELMRegressor
from random_hidden_layer import SimpleRandomHiddenLayer, RBFRandomHiddenLayer


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
    

def trainADAHERF(X, Y, dim=35):
    """
    Train AdaHERF algorithm
    """
    n_samps, NF = X.shape

    # From the 80% of training data we use 30% for ensemble model selection and 70% for real training.
    x_train, x_trainADAHERF, \
    y_train, y_trainADAHERF = cross_validation.train_test_split(X, Y, test_size=0.7)
    
    # We generate ensemble composition
    ensembleComposition = clasProbDist(x_train, y_train, dim)
    
    classifiers = []
    inforotar = []
    media = np.mean(x_trainADAHERF,axis=0)
    
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
            FK[j,v] = 1
    
        
        R = np.zeros((NF,NF))
        for l in range(0,K):
            # Let Xij be the data set X for the features in Fij
            pos = np.nonzero(FK[l,:])[0]

            vpos = [pos[m] for m in range(0, len(pos))]
   
            XXij = X[:, vpos]
            pca = apply_pca(XXij, Y, len(pos))

            COEFF = pca.components_
            
            for indI in range(0,len(pos)):
                for indJ in range(0,len(pos)):
                    R[pos[indI], pos[indJ]] = COEFF[indI,indJ]

                    
        inforotar.append(R)
        Xrot = x_trainADAHERF.dot(R) - media
        
        if ensembleComposition[i] == 0:
            #print "DT"
            dt = tree.DecisionTreeClassifier()
            dt = dt.fit(Xrot, y_trainADAHERF)
            classifiers.append(dt)
            
        if ensembleComposition[i] == 1:
            #print "ELM"
            hiddenN = 1000
            if len(y_trainADAHERF)/3 < hiddenN:
                hiddenN = len(y_trainADAHERF)/3
                
            elm = SimpleELMClassifier(n_hidden=hiddenN)
            elm.fit(Xrot, y_trainADAHERF)
            classifiers.append(elm)

    return classifiers, inforotar, media


def testADAHERF(X, classifiers, inforotar, media):   
    """
        Test ADAHERF classifier
    """
    
    dim = len(classifiers)
    row, col = X.shape
    ensembleOutput = np.zeros((row,dim))

    for i in range(0,dim):
        ensembleOutput[:,i] = classifiers[i].predict(X.dot(inforotar[i])-media)

    y_pred = mode(ensembleOutput, axis=1)[0]
    
    return y_pred
    
if __name__ == '__main__':


    # 1. Balance
    # 2. Breast-can
    # 3. Diabetes
    # 4. Ecoli
    # 5. Iris
    # 6. Liver
    # 7. Sonar
    # 8. Soybean
    # 9. Spambase
    # 10. Waveform
    # 11. Wine
    # 12. Digit
    # 13. Hayes
    # 14. Monk1
    # 15. Monk2
    # 16. Monk3

    for i in range(1,17):

        uci_path = get_uci_path()
        X, Y = read_uci_dataset(uci_path,i)
        
        # Train test split
        # We save x_test and y_test for testing => 20% of the total data.
        x_train, x_test, y_train,y_test = cross_validation.train_test_split(X, Y, test_size=0.2)
        
        classifiers, inforotar, media = trainADAHERF(x_train,y_train)
        
        # For testing x_test & y_test
        y_pred = testADAHERF(x_test, classifiers, inforotar, media)
        
        cm = confusion_matrix(y_test, y_pred)
        print "BD: ",i,"=>",float(cm.trace())/cm.sum()
