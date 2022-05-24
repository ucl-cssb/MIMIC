import numpy as np

import pandas as pd
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
from gMLV import *
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage



SMALL_SIZE = 11
MEDIUM_SIZE = 14
BIGGER_SIZE = 17

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def plot_dendrogram(model, **kwargs):
    count = np.zeros(model.children_.shape[0])
    nsamples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        currentcount = 0
        for child_idx in merge:
            if child_idx < nsamples:
                currentcount += 1
            else:
                currentcount += count[child_idx - nsamples]
        count[i] = currentcount

    linkagematrix = np.column_stack(
        [model.children_, model.distances_, count]
    ).astype(float)

    dendrogram(linkagematrix, **kwargs)

Y = np.loadtxt(open("./data/Y.csv", "rb"), delimiter=",")
F = np.loadtxt(open("./data/F.csv", "rb"), delimiter=",")

Y = Y.T

print(Y.shape)

pca = PCA(n_components=107, svd_solver="randomized", whiten=True).fit(Y)

plt.plot(range(1,108), np.cumsum(pca.explained_variance_ratio_))
plt.ylim(bottom = 0.0)
plt.xlabel('Number of dimensions')
plt.ylabel('Explained variance')
plt.savefig('cum_var.png', dpi=300)


n_components = 2
pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True)

Y_pca = pca.fit_transform(Y)

print(np.sum(pca.explained_variance_ratio_))
print(Y_pca.shape)

k_means = 0
spectral = 0
guassian_mixture = 0
hierarchical = 1

if k_means:
    n_clusters = 3
    km = KMeans(
        n_clusters=n_clusters, init='random',
        n_init=1000, max_iter=30000,
        tol=1e-010
    )
    c_km = km.fit_predict(Y_pca)
    print(c_km.shape)

if spectral:
    n_clusters = 3
    km = SpectralClustering(n_clusters=3,
      assign_labels='discretize', affinity='nearest_neighbors')

    c_km = km.fit_predict(Y_pca)
    print(c_km.shape)


if guassian_mixture:
    n_clusters = 3
    km = GaussianMixture(n_components=n_clusters, covariance_type = 'full', max_iter = 10000, n_init = 10, init_params = 'random', verbose = 1, tol = 1e-10)

    c_km = km.fit_predict(Y_pca)
    print(c_km.shape)

if hierarchical:
    n_clusters = 3
    km = AgglomerativeClustering(n_clusters = n_clusters, linkage = 'ward', compute_distances = True)

    c_km = km.fit_predict(Y_pca)
    print(c_km.shape)
    plt.figure()
    plt.title("Hierarchical Clustering Dendrogram")
    plot_dendrogram(km, truncate_mode="level", p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")


plt.figure()
for i in range(n_clusters):
    plt.scatter(
        Y_pca[c_km == i, 0], Y_pca[c_km == i, 1],
        label='cluster ' + str(i)
    )
plt.savefig('clustering.png', dpi = 300)
plt.show()