import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_analysis'))

from gMLV import *
from load_data import *

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.patches import FancyArrowPatch
from collections import Counter
from mpl_toolkits.mplot3d import proj3d

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


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

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

data = load_subject_ID_dict()
X, subjects = get_X_subjects(data) # extract the taxa data ordered by subject and time
perts = load_perturbations_dict()

combined_dict = combine_taxa_pert_dicts(data, perts)

species_indicators = []

for key in combined_dict.keys():


    dat = combined_dict[key]
    s_i = np.zeros(len(dat[0][1])) == 1

    for d in dat:
        taxa = d[1]

        s_i = s_i | taxa > 0

    species_indicators.append(s_i)

print(len(dat[0][1]))
print(len(combined_dict.keys()))
species_indicators = np.array(species_indicators)
print(species_indicators.shape) # (n_subjects, n_taxa)
species_counts = np.sum(species_indicators, axis = 1)
print(species_counts)
print(np.sum(species_indicators, axis = 0))


plt.bar(range(len(species_counts)), sorted(species_counts, reverse = True))
plt.xlabel('subject')
plt.ylabel('species count')

plt.figure()
plt.bar(range(len(np.sum(species_indicators, axis = 0))), sorted(np.sum(species_indicators, axis = 0), reverse = True))
plt.xlabel('species')
plt.ylabel('subject count')
plt.show()

sys.exit()

X, P, subjects = get_X_P_subjects(combined_dict)

subject_status = get_subject_status()

all_subject_info = combine_subject_dicts(subjects, subject_status)



#print(subjects[0])
print(X.shape)
print(P.shape)
#print(combined_dict['M10'])
#print(P[subjects[0][1]:subjects[0][2]])


plt.spy(X)
plt.figure()
plt.spy(P.T)


print('shape:', X.shape)





# plt.figure()
#
# pca = PCA(n_components=94, svd_solver="randomized", whiten=True).fit(X)
# print(pca.components_[0])
# print(np.cumsum(pca.explained_variance_ratio_))
# plt.plot(range(1,95), np.cumsum(pca.explained_variance_ratio_))
# plt.ylim(bottom = 0.0)
# plt.xlabel('Number of dimensions')
# plt.ylabel('Explained variance')
# plt.savefig('cum_var.png', dpi=300)
#
# sys.exit()


n_components = 3
pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True)

X_pca = pca.fit_transform(X)

print(np.sum(pca.explained_variance_ratio_))
print(X_pca.shape)

k_means = 0
spectral = 1
guassian_mixture = 0
hierarchical = 0

if k_means:
    n_clusters = 4
    km = KMeans(
        n_clusters=n_clusters, init='random',
        n_init=1000, max_iter=30000,
        tol=1e-10
    )
    c_km = km.fit_predict(X_pca)
    print(c_km.shape)

if spectral:
    n_clusters = 4
    km = SpectralClustering(n_clusters=n_clusters,
      assign_labels='discretize', affinity='nearest_neighbors', n_neighbors = 10, random_state=1)

    c_km = km.fit_predict(X_pca)
    print(c_km.shape)


if guassian_mixture:
    n_clusters = 4
    km = GaussianMixture(n_components=n_clusters, covariance_type = 'full', max_iter = 10000, n_init = 10, init_params = 'random', verbose = 1, tol = 1e-10)

    c_km = km.fit_predict(X_pca)
    print(c_km.shape)

if hierarchical:
    n_clusters = 4
    km = AgglomerativeClustering(n_clusters = n_clusters, linkage = 'ward', compute_distances = True)

    c_km = km.fit_predict(X_pca)
    print(c_km.shape)
    plt.figure()
    plt.title("Hierarchical Clustering Dendrogram")
    plot_dendrogram(km, truncate_mode="level", p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")




def plot_patient_cluster_counts():
    subject_clusters = [] # unknown status
    h_s_clusters = [] # healthy survived
    d_s_clusters = [] # diseased survived
    d_d_clusters = [] # diseased survived

    for key in all_subject_info:
        subject = all_subject_info[key]
        print(subject)
        clusters = c_km[subject['start']:subject['end']]


        cluster_flags = [int(i in clusters) for i in range(4)]

        decimal = 0
        for digit in cluster_flags:
            decimal = decimal * 2 + int(digit)

        if subject['class'] == 'healthy' and subject['outcome'] == 'survival':
            h_s_clusters.append(decimal)
        elif subject['class'] == 'disease' and subject['outcome'] == 'survival':
            d_s_clusters.append(decimal)
        elif subject['class'] == 'disease' and subject['outcome'] == 'mortality':
            d_d_clusters.append(decimal)
        else:
            subject_clusters.append(decimal)


    #plot patients by cluster visiting
    plt.close('all')
    plt.figure()


    counts = Counter(subject_clusters)
    h_s_counts = Counter(h_s_clusters)
    d_s_counts = Counter(d_s_clusters)
    d_d_counts = Counter(d_d_clusters)

    counters = [counts, h_s_counts, d_s_counts, d_d_counts]
    for i in range(16):

        for counter in counters:
            if i not in counter.keys():
                counter[i] = 0


    binary = lambda i: format(i, '04b')

    labels = ['unknown', 'healthy, survived', 'disease, survived', 'diseased, died']
    width = 0.15
    for i, counter in enumerate(counters):

        print(counter.keys())
        x = np.array(list(counter.keys())) + (i-2)*width
        print(x)
        if i == 2:
            plt.bar(x, counter.values(), width=width, tick_label=list(map(binary, counter.keys())), label=labels[i],
                    align='edge')
        else:
            plt.bar(x, counter.values(), width=width,label=labels[i],
                    align='edge')

    plt.xlabel('Clusters visited')
    plt.ylabel('Number of patients')
    plt.legend()
    plt.show()


def plot_cluster_status_counts():
    # count the status of each data point within each cluster

    cluster_counts = np.zeros((4, n_clusters))
    for subject_ID in all_subject_info:
        subject = all_subject_info[subject_ID]

        subject_clusters = np.array([0]*n_clusters)

        for i in range(n_clusters):

            subject_clusters[i] = np.sum(c_km[subject['start']:subject['end']] == i)

        print()
        print(cluster_counts)
        if subject['class'] == 'healthy' and subject['outcome'] == 'survival':

            cluster_counts[1,:] += subject_clusters


        elif subject['class'] == 'disease' and subject['outcome'] == 'survival':

            cluster_counts[2,:] += subject_clusters

        elif subject['class'] == 'disease' and subject['outcome'] == 'mortality':

            cluster_counts[3,:] += subject_clusters
        else:

            cluster_counts[0,:] += subject_clusters



    plt.figure()
    lables = ['unknown', 'healthy, survived', 'disease, survived', 'disease, died']
    for i in range(0,4):
        width = 0.2
        x = np.arange(n_clusters) + width*i
        h = cluster_counts[i,:]
        plt.bar(x, h, width = width, tick_label=np.arange(n_clusters), label = lables[i])

    plt.legend()
    plt.xlabel('Cluster')
    plt.ylabel('Sample count')

def plot_by_cluster():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')



    #plot by cluster
    for i in range(n_clusters):
        ax.scatter(
            X_pca[c_km == i, 0], X_pca[c_km == i, 1],X_pca[c_km == i, 2],
            label='cluster ' + str(i), marker = '.'
        )

    plt.legend()



def plot_by_status():
    # plot by status
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    count = 0
    for subject_ID in all_subject_info:
        subject = all_subject_info[subject_ID]


        if subject['class'] == 'healthy' and subject['outcome'] == 'survival':
            ax.scatter(
                    X_pca[subject['start']:subject['end'], 0], X_pca[subject['start']:subject['end'], 1],X_pca[subject['start']:subject['end'], 2],
                    color = 'black', marker = '.'
                )
            count += len(X_pca[subject['start']:subject['end'], 0])
        elif subject['class'] == 'healthy' and subject['outcome'] == 'mortality':
            ax.scatter(
                X_pca[subject['start']:subject['end'], 0], X_pca[subject['start']:subject['end'], 1],
                X_pca[subject['start']:subject['end'], 2],
                color='red', marker='.'
            )
            count += len(X_pca[subject['start']:subject['end'], 0])
        elif subject['class'] == 'disease' and subject['outcome'] == 'survival':
            ax.scatter(
                X_pca[subject['start']:subject['end'], 0], X_pca[subject['start']:subject['end'], 1],
                X_pca[subject['start']:subject['end'], 2],
                color='green', marker='.'
            )
            count += len(X_pca[subject['start']:subject['end'], 0])
        elif subject['class'] == 'disease' and subject['outcome'] == 'mortality':
            ax.scatter(
                X_pca[subject['start']:subject['end'], 0], X_pca[subject['start']:subject['end'], 1],
                X_pca[subject['start']:subject['end'], 2],
                color='orange', marker='.'
            )
            count += len(X_pca[subject['start']:subject['end'], 0])
        else:
            ax.scatter(
                X_pca[subject['start']:subject['end'], 0], X_pca[subject['start']:subject['end'], 1],
                X_pca[subject['start']:subject['end'], 2],
                color='blue', marker=''
            )



def plot_perturbations():
    # plot the perturbations


    colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'black']
    count = 0

    unique_perts = {str(P[0]):0}

    for i, p in enumerate(P):

        if str(p) not in unique_perts.keys():
            unique_perts[str(p)] = len(unique_perts.keys())

        if sum(p)>0:
            a = Arrow3D(X_pca[i:i+2,0],X_pca[i:i+2,1], X_pca[i:i+2,2] , arrowstyle='-|>', mutation_scale=10, linestyle = ':', color = colours[unique_perts[str(p)]])
            ax.add_artist(a)


    print()
    print(unique_perts)


    subject = subjects[1]
    print(c_km[subject[1]:subject[2]])

def plot_traj(ax, X, start, end, colour):
    '''
    plots the subjects trajectory with arrows to indicate direction at each timestep
    '''
    ax.plot(X[start:end, 0], X[start:end, 1],
            X[start:end, 2],
            color=colour)

    # plot arrows
    for i in range(start, end):
        a = Arrow3D(X[i:i + 2, 0], X[i:i + 2, 1], X[i:i + 2, 2], arrowstyle='-|>', mutation_scale=10,
                    color=colour)
        ax.add_artist(a)

def plot_labelled_trajectories():
    for subject_ID in all_subject_info:

        subject = all_subject_info[subject_ID]

        if subject['class'] == 'healthy' and subject['outcome'] == 'survival':
            plot_traj(ax, X_pca, subject['start'], subject['end'], 'black')
        if subject['class'] == 'healthy' and subject['outcome'] == 'mortality':
            plot_traj(ax, X_pca, subject['start'], subject['end'], 'red')
        if subject['class'] == 'disease' and subject['outcome'] == 'survival':
            plot_traj(ax, X_pca, subject['start'], subject['end'], 'green')
        if subject['class'] == 'disease' and subject['outcome'] == 'mortality':
            plot_traj(ax, X_pca, subject['start'], subject['end'], 'orange')


    #ax.plot(X_pca[:, 0][subject[1]:subject[2]], X_pca[:, 1][subject[1]:subject[2]], X_pca[:, 2][subject[1]:subject[2]], color = 'black')
    plt.legend()
    ax.set_xlabel('D1')
    ax.set_ylabel('D2')
    ax.set_zlabel('D3')
    plt.show()
    plt.savefig('clustering.png', dpi = 300)
