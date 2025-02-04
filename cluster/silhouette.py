import numpy as np
from scipy.spatial.distance import cdist
from utils import make_clusters
from kmeans import KMeans

class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`


        Silhouette Coefficient for a sample is:
        (b - a) / max(a, b)
        where a is how far that sample is from other samples in the same cluster (on average)
        and b is how far the smallest mean distance to a different cluster
        """

        # checking for input errors
        if type(X) != np.ndarray:
            raise TypeError("X must be a numpy array")
        if type(y) != np.ndarray:
            raise TypeError("y must be a numpy array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of observations")
        if len(y.shape) != 1:
            raise ValueError("y must be a 1D array")
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D matrix")

        # calculate the silhouette score for each of the observations
        scores = []

        for i in range(X.shape[0]):
            cluster_ix = y[i]
            # print("X[i]:", X[i])
            # print("X[y == cluster_ix]:", X[y == cluster_ix])
            a = np.mean(cdist(X[y == cluster_ix], [X[i]]))

            # print("--", cluster_ix, a)
            dist_from_cluster = cdist([X[i]], X[y != cluster_ix])

            # print("----", dist_from_cluster)
            b = np.min(dist_from_cluster)

            scores.append((b - a) / np.max([a, b]))

        return np.array(scores)