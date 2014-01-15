#!/usr/bin/python

import os
import numpy as np
from sklearn.decomposition import PCA
from time import time
from sklearn.cluster import k_means
from elm import ELMClassifier, ELMRegressor, SimpleELMClassifier, SimpleELMRegressor
from random_hidden_layer import SimpleRandomHiddenLayer, RBFRandomHiddenLayer
from sklearn import cross_validation
from sklearn import tree

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

#Desde arriba, hasta aqui, lo copias y lo pegas al ipy, a parte.

#esto se pone para cuando este fichero se
#va a ejecutar como un script, desde fuera
#Lo que hay aqui debajo lo puedes poner en otro sitio.
if __name__ == '__main__':

    uci_path = get_uci_path()
	data, labels = read_uci_dataset(uci_path)
	n_samps, n_dim = data.shape
	pca = apply_pca(data, labels, n_dim)
	print(pca.components_) 

    # pca.components_ is the rotation matrix.
    rotate = data.dot(pca.components_)
	
	# We are going to try ELM
	x_train, x_test, y_train,y_test = cross_validation.train_test_split(data, labels, test_size=0.2)
	elmc = SimpleELMClassifier(n_hidden=1000)
	elmc.fit(x_train, y_train)
	print elmc.score(x_test, y_test)
	
	# Decision tree
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(x_train, y_train)
	clf.predict(x_test)

