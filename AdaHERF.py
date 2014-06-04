# Authors: Borja Ayerdi [borja.ayerdi -at- ehu -dot- com] 
#          Alexandre Manhaes Savio [alexandre.manhaes -at- ehu -dot- com]
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
import random
import numpy as np

from scipy.stats import mode

from sklearn import tree
from sklearn import cross_validation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from elm import SimpleELMClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.qda import QDA


__all__ = ["AdaHERF"]


class AdaHERF(object):
    """
    AdaHERF
    """

    def __init__(self, n_classifiers=35):

        self._n_classifiers = n_classifiers
        self._classifiers = []
        self._inforotar = []
        self._media = None
        self._scaler = StandardScaler()
        self._std = []
        self._med = []
        self._noise = []
        self._listclassifiers = [
            DecisionTreeClassifier,
            SimpleELMClassifier,
            KNeighborsClassifier,
            LinearSVC,
            RandomForestClassifier,
            AdaBoostClassifier,
            GaussianNB]

    @staticmethod
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

    def _clasProbDist(self,data,labels):
        """
        Will return a vector with the composition, type of classifiers, of the ensemble.
        """
        
        #rankscore =[55,34,21,13,8,5,3,2,1,1] # Fibbonacci
        rankscore = [13,8,5,3,2,1,1] # Fibbonacci
        
        # Train test split
        x_train, x_test, \
        y_train, y_test = cross_validation.train_test_split(data, labels, test_size=0.2)
        
        # Matrix to order
        matrix = []
        
        for code, cl in zip(range(0,len(self._listclassifiers)),self._listclassifiers):
            cl = cl()
            cl.fit(x_train, y_train)
            matrix.append([cl.score(x_test, y_test),code])
        
        # Sort with the error of the classification
        matrix.sort()
        matrix.reverse()
        
        probDist = []
        for i in range(0,len(matrix)):
            a = matrix[i]
            a = a[1]
            probDist.extend(np.tile(a, rankscore[i]))
            
        ensembleComposition = []
        for j in range(0,self._n_classifiers):
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
        
        # Compute mean, std and noise for z-score
        self._std = np.std(X,axis=0)
        self._med = np.mean(X,axis=0)
        self._noise = [random.uniform(-0.000005, 0.000005) for p in range(0,X.shape[1])]
        
        # Apply Z-score
        Xz = (X-self._med)/(self._std+self._noise)
        
        # From the 80% of training data we use 30% for ensemble model selection and 70% for real training.
        x_train, x_trainADAHERF, \
        y_train, y_trainADAHERF = cross_validation.train_test_split(Xz, Y, test_size=0.7)
        
        # We generate ensemble composition
        ensembleComposition = self._clasProbDist(x_train, y_train)
        
        for i in range(0,self._n_classifiers):
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
                # Let Xzij be the data set X for the features in Fij
                pos = np.nonzero(FK[l,:])[0]

                vpos = [pos[m] for m in range(0, len(pos))]
       
                Xzij = Xz[:, vpos]
                pca = self._apply_pca(Xzij, Y, len(pos))

                for indI in range(0,len(pca.components_)):
                    for indJ in range(0,len(pca.components_)):
                        R[pos[indI], pos[indJ]] = pca.components_[indI,indJ]

                        
            self._inforotar.append(R)
            Xrot = x_trainADAHERF.dot(R)
            
            cl = self._listclassifiers[ensembleComposition[i]]()
            cl.fit(Xrot, y_trainADAHERF)
            self._classifiers.append(cl)

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
        ensemble_output = np.zeros((len(X),dim))
        
        # Z-score
        X = (X-self._med)/(self._std+self._noise)

        for i in range(0,dim):
            xrot_z = X.dot(self._inforotar[i])
            ensemble_output[:,i] = self._classifiers[i].predict(xrot_z)

        y_pred = mode(ensemble_output, axis=1)[0]
        
        return y_pred
