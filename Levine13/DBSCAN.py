# Importing required modules


from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
pca_dim = 2
import numpy as np
# Load Data
data = pd.read_csv("Levine_13dim.txt", sep="\t")
print(data.head())
data = data.dropna()
labels = data['label']
print(labels.shape)
data = data.drop(['label'], axis=1)
print(data.shape)
from sklearn.cluster import DBSCAN

# Transform the data
pca = PCA(pca_dim)
df = pca.fit_transform(data)

db = DBSCAN(eps=0.1, min_samples=10).fit(df)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)


# #############################################################################
# Plot result


# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = df[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = df[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()
