# import the necessary packages
import os
from operator import itemgetter

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def get_train_data(_data_path):
    f_name = "{}/labels.csv".format(_data_path)
    labels = pd.read_csv(f_name, header=None).as_matrix()[:, 1]
    labels = list(map(itemgetter(1),
                      map(os.path.split,
                          map(os.path.dirname, labels))))  # Get the Directory
    le = LabelEncoder().fit(labels)
    labels_num = le.transform(labels)
    n_classes = len(le.classes_)
    f_name = "{}/reps.csv".format(_data_path)
    embeddings = pd.read_csv(f_name, header=None).as_matrix()
    return n_classes, labels_num, embeddings


if __name__ == '__main__':
    total_class, y, X = get_train_data('/media/abhi/openface-vol/embeddings')
    x_small = X[:]
    # Standardizing the features
    x_small = StandardScaler().fit_transform(x_small)
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x_small)
    # print(principalComponents[0])

    # Apply Clustering Algorithms
    # Mean Shift Clustering
    # The following bandwidth can be automatically detected using
    # bandwidth = estimate_bandwidth(principalComponents, quantile=0.3, n_samples=150)
    #
    # ms = MeanShift(bandwidth=bandwidth, bin_seeding=False, cluster_all=True)
    # ms.fit(principalComponents)
    #
    # labels = ms.labels_
    # cluster_centers = ms.cluster_centers_
    #
    # labels_unique = np.unique(labels)
    # n_clusters_ = len(labels_unique)
    #
    # print("number of estimated clusters : %d" % n_clusters_)

    # K-means Clustering
    random_state = 170
    kmeans = KMeans(n_clusters=9, random_state=random_state)
    y_pred = kmeans.fit_predict(principalComponents)
    accuracy = accuracy_score(y[:], y_pred)
    print("accuracy: {}".format(accuracy))

    # Plot Our Data
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'lime', 'violet']

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    for idx, point in enumerate(principalComponents):
        ax.scatter(point[0], point[1], point[2], c=colors[y_pred[idx]])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    bx = fig.add_subplot(1, 2, 2, projection='3d')

    for idx, point in enumerate(principalComponents):
        bx.scatter(point[0], point[1], point[2], c=colors[y[idx]])

    bx.set_xlabel('X Label')
    bx.set_ylabel('Y Label')
    bx.set_zlabel('Z Label')

    plt.show()

