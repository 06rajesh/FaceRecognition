# import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from FaceClassifier import FaceClassifier

if __name__ == '__main__':
    clf = FaceClassifier('/media/embeddings/myClassifier.pkl')
    n_classes, y, X = clf.get_train_data('/media/embeddings')
    # print(X.shape)
    # Standardizing the features
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    # print(principalComponents.shape)
    # plt.plot([1, 2, 3, 4])
    # plt.ylabel('some numbers')
    # plt.show()
