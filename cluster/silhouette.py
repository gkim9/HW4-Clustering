import numpy as np
from scipy.spatial.distance import cdist

class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray, centroid_list=None) -> np.ndarray:
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
         => meaning that b is the mean distance from that point to all other points in the cluster closest to it
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

            # Calculating the average distance from the point of interest to other points in the same cluster
            a = np.mean(cdist(X[y == cluster_ix], [X[i]], metric='euclidean'))

            # identifying the next closest cluster by calculating the distance between cluster and centroids
            next_closest_cluster = cdist(centroid_list, [X[i]])

            # replacing the point's actual cluster with +inf so that we can get the next minimum
            next_closest_cluster[cluster_ix] = float("+inf")
            next_cluster_ix = np.argmin(next_closest_cluster)

            # calculating the distance from the point to all points in the next closest cluster
            dist_from_cluster = cdist([X[i]], X[y == next_cluster_ix], metric='euclidean')
            b = np.mean(dist_from_cluster)

            # calculating score based on formula
            score = (b - a) / np.max([a, b])
            scores.append(score)

        return np.array(scores)