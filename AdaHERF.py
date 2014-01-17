# Authors: Borja Ayerdi [borja.ayerdi -at- ehu -dot- com] || Alexandre [ -at- ehu -dot- com]
# Copyright(c) 2014
# License: Simple BSD

"""
This module implements Adaptative Hybrid Extreme Rotation Forest (adaHERF)

References
----------
.. [1] Borja Ayerdi, Manuel Grana. "Hybrid Extreme Rotation Forest",
          Neural Networks, 2014.
.. [2] http://www.extreme-learning-machines.org
.. [3] G.-B. Huang, Q.-Y. Zhu and C.-K. Siew, "Extreme Learning Machine:
          Theory and Applications", Neurocomputing, vol. 70, pp. 489-501,
          2006.
.. [4] Fernandez-Navarro, et al, "MELM-GRBF: a modified version of the  
          extreme learning machine for generalized radial basis function  
          neural networks", Neurocomputing 74 (2011), 2502-2510

"""

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

__all__ = ["AdaHERF"]


class AdaHERF(X, Y):
    """
    AdaHERF
    """

    def __init__(self, n_classifiers=35):

        self._n_classifiers = n_classifiers
        self._classifiers = []
        self._inforotar = []
        self._media = None

    def _apply_pca(data, labels, n_comps=1):
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

    def _clasProbDist(data,labels,dim):
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
        
        # Train/Test ELM == 1
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

    def fit(self, X, Y):
        """
        Fit the model using X, y as training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]
        Training vectors, where n_samples is the number of samples
        and n_features is the number of features.

        y : array-like of shape [n_samples, n_outputs]
        Target values (class labels in classification, real numbers in
        regression)
 
        Returns
        -------
        self : object

        Returns an instance of self.
        """
        n_samps, NF = X.shape

        # From the 80% of training data we use 30% for ensemble model selection and 70% for real training.
        x_train, x_trainADAHERF, \
        y_train, y_trainADAHERF = cross_validation.train_test_split(X, Y, test_size=0.7)
        
        # We generate ensemble composition
        ensembleComposition = self._clasProbDist(x_train, y_train, dim)
        
        self._media = np.mean(x_trainADAHERF,axis=0)
        
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
                pca = self._apply_pca(XXij, Y, len(pos))

                for indI in range(0,len(pos)):
                    for indJ in range(0,len(pos)):
                        R[pos[indI], pos[indJ]] = pca.components_[indI,indJ]

                        
            self._inforotar.append(R)
            Xrot = x_trainADAHERF.dot(R) - self._media
            
            if ensembleComposition[i] == 0:
                #print "DT"
                dt = tree.DecisionTreeClassifier()
                dt = dt.fit(Xrot, y_trainADAHERF)
                self._classifiers.append(dt)
                
            if ensembleComposition[i] == 1:
                #print "ELM"
                hiddenN = 1000
                if len(y_trainADAHERF)/3 < hiddenN:
                    hiddenN = len(y_trainADAHERF)/3
                    
                elm = SimpleELMClassifier(n_hidden=hiddenN)
                elm.fit(Xrot, y_trainADAHERF)
                self._classifiers.append(elm)

        return self

    def predict(self, X):
        """
        Predict values using the model

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]

        Returns
        -------
        C : numpy array of shape [n_samples, n_outputs]
            Predicted values.
        """
        dim = len(self._classifiers)
        ensemble_ouput = np.zeros_like(X)

        for i in range(0,dim):
            ensemble_output[:,i] = classifiers[i].predict(X.dot(self._inforotar[i]) - self._media)

        y_pred = mode(ensemble_output, axis=1)[0]
        
        return y_pred
