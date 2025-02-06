# write your silhouette score unit tests here
import numpy as np
import pytest
from sklearn import metrics as skmetrics

from cluster import KMeans, make_clusters, Silhouette

s = Silhouette()
X, labels = make_clusters(n=100, k=2)
km = KMeans(2)
km.fit(X)
y = km.predict(X)
centroid_list = km.get_centroids()

def test_wrong_input():
    # wrong type of input for X
    with pytest.raises(TypeError):
        s.score(X='string', y=np.array([1, 1, 1]), centroid_list=[0,0])

    # wrong type of input for y
    with pytest.raises(TypeError):
        s.score(X=X, y='string', centroid_list=[0,0])

    # testing different # of points in X and y
    with pytest.raises(ValueError):
        s.score(X=X, y=np.array([1, 1, 1]), centroid_list=[0,0])

    # testing non-1D matrix of y
    with pytest.raises(ValueError):
        s.score(X=X, y=np.array([[1, 1, 1], [2, 2, 2]]), centroid_list=[0,0])

    # testing non-2D matrix of X
    with pytest.raises(ValueError):
        s.score(X=np.array([1, 1, 1]), y=y, centroid_list=[0,0])

def test_Silhouette():
    '''
    Comparing my Silhouette score with scikit-learn's score

    taking the average difference between the two scoring metrics
    '''

    acceptable_error = 0.1

    # generating Silhouette score for all samples using sci-kit learn's metrics
    sk_score = skmetrics.silhouette_samples(X, y)

    # generating Silhouette score for all samples 
    my_score = s.score(X, y, centroid_list)

    # genetrating the difference in sklearn and my Silhouette score for all samples
    diff = sk_score-my_score

    # take the absolute value of the differences of all samples and take the mean
    avg_score_diff = np.mean(abs(diff))
    
    assert avg_score_diff <= acceptable_error, "Score between silhouette.py and sklearn's silhouette_score is too different"