# import the necessary packages
import os
from operator import itemgetter

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering
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


def get_diff(img1, img2):
    d = img1 - img2
    return np.dot(d, d)


if __name__ == '__main__':
    total_class, y, X = get_train_data('/media/abhi/openface-vol/embeddings')
    batch = 150
    x_small = X[0:batch]

    # idx = 0
    # while y[idx] == 3:
    #     n1 = idx
    #     n2 = idx + 1
    #     print('{} - {}, {} : {}'.format(idx, y[n1], y[n2], get_diff(X[n1], X[n2])))
    #     idx += 1

    x_small = StandardScaler().fit_transform(x_small)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x_small)
    clustering = AgglomerativeClustering(linkage='ward', n_clusters=3)
    clustering.fit(x_small)
    print(clustering.labels_)
