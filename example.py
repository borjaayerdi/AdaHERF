#!/usr/bin/python

import os
import time
import random
import numpy as np

from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import StratifiedKFold

from AdaHERF import AdaHERF

def get_uci_path():
    """
    Returns the path to the UCI datasets, depending on the 
    host name.

    :return: string
    """
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
    #for i in range(1,2):

        uci_path = get_uci_path()
        X, Y = read_uci_dataset(uci_path,i)
        
        K = 10
        vAcc = []
        cv = StratifiedKFold(Y, K)
        
        for train, test in cv:
        
            x_train, x_test, y_train, y_test = X[train,:], X[test,:], Y[train], Y[test]
            # Train test split
            # We save x_test and y_test for testing => 20% of the total data.
            #x_train, x_test, y_train,y_test = cross_validation.train_test_split(X, Y, test_size=0.2)
          
            adaherf = AdaHERF()
            adaherf = adaherf.fit(x_train, y_train)

            # For testing x_test & y_test
            y_pred = adaherf.predict(x_test)
            
            cm = confusion_matrix(y_test, y_pred)
            acc = float(cm.trace())/cm.sum()
            vAcc.append(acc)
            #print "BD: ",i,"=>",acc

        bd_std = np.std(vAcc)
        bd_acc = np.mean(vAcc)
        print format(bd_acc*100,'.2f'), format(bd_std*100,'.2f')