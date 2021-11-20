# Importing required modules

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
pca_dim = 3
# Load Data
data = pd.read_csv("Levine_13dim.txt", sep="\t")
labels = data['label'].dropna()
data = data.dropna()



from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(data.values)
print(X_embedded.shape)