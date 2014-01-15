import numpy as np
from sklearn.decomposition import PCA

# Load the data
dataset = 1
ruta = ''.join(['uci-datasets/', str(dataset), '.data']) 
data = np.genfromtxt(ruta,delimiter=",")
col = len(data[0]) # In python we start in 0
fil = len(data) # In python we start in 0

# Delete the first column of labels
Y = data[:,0]
X = data[:,1:col]
col = col -1

# PCA
pca = PCA(n_components=col)
pca.fit(X)
PCA(copy=True, n_components=col, whiten=False)
print(pca.components_) 

# pca.components_ is the rotation matrix.
rotate = X.dot(pca.components_)

