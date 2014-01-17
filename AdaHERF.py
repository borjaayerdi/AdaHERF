# Author: Borja Ayerdi [borja.ayerdi -at- ehu -dot- com]
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

__all__ = ["adaHERF"]


class AdaHERF(X, Y):
    """
    AdaHERF
    """

    def __init__(self, n_classifiers=35):

        self._n_classifiers = n_classifiers



    def decision_function(self, X):
        """
        This function return the decision function values related to each
        class on an array of test vectors X.

        Parameters
        ----------
        X : array-like of shape [n_samples, n_features]

        Returns
        -------
        C : array of shape [n_samples, n_classes] or [n_samples,]
            Decision function values related to each class, per sample.
            In the two-class case, the shape is [n_samples,]
        """
        return self.elm_classifier_.decision_function(X)

    def fit(self, X, y):
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
        rhl = SimpleRandomHiddenLayer(n_hidden=self.n_hidden,
                                      activation_func=self.activation_func,
                                      activation_args=self.activation_args,
                                      random_state=self.random_state)

        self.elm_classifier_ = ELMClassifier(hidden_layer=rhl)
        self.elm_classifier_.fit(X, y)

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
        if (self.elm_classifier_ is None):
            raise ValueError("SimpleELMClassifier not fitted")

        return self.elm_classifier_.predict(X)
