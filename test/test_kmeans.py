# Write your k-means unit tests here
import numpy as np
import pytest

from cluster import KMeans, make_clusters

k = 3
mat, label = make_clusters(k=k)

def test_invalid_KMeans_initialization():
    '''
    k need to be an integer greater than 0
    max_iter need to be an integer greater than 0
    tol need to be a float greater than 0
    '''

    # testing wrong inputs of k
    with pytest.raises(TypeError):
        KMeans(k='string')
    
    with pytest.raises(TypeError):
        KMeans(k=0.5)

    with pytest.raises(ValueError):
        KMeans(k=-1)

    # testing wrong inputs of max_iter
    with pytest.raises(TypeError):
        KMeans(k=2, max_iter='string')
    
    with pytest.raises(TypeError):
        KMeans(k=2, max_iter=0.5)

    with pytest.raises(ValueError):
        KMeans(k=2, max_iter=-1)

    # testing wrong inputs of tol
    with pytest.raises(TypeError):
        KMeans(k=2, tol='string')

    with pytest.raises(TypeError):
        KMeans(k=2, tol=1)

    with pytest.raises(ValueError):
        KMeans(k=2, tol=-1.1)    

def test_invalid_mat_type():
    '''
    Testing if different wrong mat inputs will raise an error
    1. Test if mat is not a numpy array
    2. Test if mat is not a 2D matrix
    '''
    kmeans = KMeans(k=2)

    with pytest.raises(TypeError):
        kmeans.fit(mat='string')
    
    with pytest.raises(Exception):
        kmeans.fit(mat=np.array([0, 0, 0]))


def test_different_mat_input():
    '''
    Testing if using different matrix for fit vs predict will raise an error
    '''
    kmeans = KMeans(k=2)

    mat = np.array([[0, 0], [1, 1]])
    mat2 = np.array([[0, 0], [1, 1], [2, 2]]) # testing different size of array
    mat3 = np.array([[1, 1], [2, 2]]) # testing different values of array
    mat4 = np.array([[1,1], [0, 0]]) # will also give error because it is not identical to mat-- not sure if we want this to throw an error, though

    kmeans.fit(mat=mat)

    with pytest.raises(ValueError):
        kmeans.predict(mat=mat2)

    with pytest.raises(ValueError):
        kmeans.predict(mat=mat3)

    with pytest.raises(ValueError):
        kmeans.predict(mat=mat4)

def test_pred():
    '''
    Running pred before fit should raise an Attribute Error
    '''
    kmeans = KMeans(k=k)
    with pytest.raises(AttributeError):
        kmeans.predict(mat)

def test_fit():
    '''
    Testing the kmeans with make clusters using small spacing (low spread amongst clusters) and see if the # points stay constant from generation to fitted model
    '''
    k = 3

    # choosing a small scale/stdev to test the fit to guarantee the KMeans will return correct designation
    mat, labels = make_clusters(n=500, k=k, scale=0.3) 

    # fitting the matrix
    my_Kmeans = KMeans(k=k)
    my_Kmeans.fit(mat)
    pred_label = my_Kmeans.predict(mat)

    # initializing dictionary to contain counts of each label
    pred_label_count = {ix:0 for ix in range(k)}
    label_count = {ix:0 for ix in range(k)}

    for label in pred_label:
        pred_label_count[label] += 1

    for label in labels:
        label_count[label] += 1

    # make sure the # of labels in each cluster is equivalent no matter the specific label designation
    assert sorted(pred_label_count.values()) == sorted(label_count.values()), "Fit function did not predict the correct labels"