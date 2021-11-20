import dexplot as dxp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Levine_13dim.txt", sep="\t")
label = data['label']
data = data.drop(['label'], axis=1)

X = data.values

from sklearn import preprocessing

std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_scaled)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())
#data.plot()

# projeter X sur les composantes principales
X_projected = pca.transform(X_scaled)
print((X_projected.shape))
# afficher chaque observation
plt.scatter(X_projected[:, 0], X_projected[:, 1],
    # colorer en utilisant la variable 'Rank'
    c=data.get('label'))

plt.colorbar()
plt.show()


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=9, random_state=0).fit(X_projected)
plt.plot(kmeans)
plt.show()


#data_norm = (data - data.mean()) / (data.max() - data.min())
#data_norm.hist(by='label')
#plt.show()


#hist = data.hist(bins=1)
#plt.plot(hist)
#plt.show()



"""
from pycaret.clustering import *
clf1 = setup(data = data, pca = True, pca_components = 2, normalize=True)
# creating a model
kmeans3 = create_model('kmeans', num_clusters = 3)
#kmeans4 = create_model('kmeans', num_clusters = 4)
# plotting a model
plot_model(kmeans3)
plt.show()
#plot_model(kmeans4)
"""