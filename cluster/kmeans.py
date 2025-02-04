import numpy as np
from scipy.spatial.distance import cdist
from utils import make_clusters

class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        if type(k) == int:
            if k > 0:
                self.num_centroid = k
            else:
                raise ValueError("k must be greater than 0")
        else:
            raise TypeError("k must be an integer")
        
        if type(tol) == float:
            if tol > 0:
                self.tolerance = tol
            else:
                raise ValueError("tol must be greater than 0")
        else:
            raise TypeError("tol must be a float")
        
        if type(max_iter) == int:
            if max_iter > 0:
                self.max_iter = max_iter
            else:
                raise ValueError("max_iter must be greater than 0")
        else:
            raise TypeError("max_iter must be an integer")
        
        # initializing attributes
        self.mat = None
        self.new_centroids = None
        self.centroids = None
        self.closest_centroid_mat = None
        self.error = None

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        # saving the input matrix as an attribute to access in the error function
        self.mat = mat

        if type(mat) != np.ndarray:
            raise TypeError("mat must be a numpy array")
        if len(mat.shape) != 2:
            raise Exception("mat must be a 2D matrix")
        
        '''
        Lloyd's algorithm provided in class:

        Given k (number of clusters) and epsilon (tolerance),
        1. Randomly initialize k cluster centroids (random values, assign to "m_0")
            a. Random values = random points from the dataset
        2. For each point in the dataset, assign it to the nearest centroid
        3. Update the centroid to be the mean of all the points assigned to it (m_1 for first iteration)
        4. Repeat steps 2 and 3 until the centroids do not change by more than epsilon (until m_i - m_{i-1} < epsilon)
        
        '''
        
        self.max_iter = 100

        # randomly initialize k cluster centroids from points in the dataset# initialize the centroids dictionary with random points from the dataset
        self.centroids = {centroid_index: mat[np.random.randint(0, mat.shape[0], size=1)] for centroid_index in range(self.num_centroid)} 
        print(self.centroids)
        
        for _ in range(self.max_iter): # only run the loop for a maximum of max_iter times
            print(self.centroids)
             # initialize the new centroid dictionary with empty lists that will be updated with the new centroids
            self.new_centroids = {centroid_index: mat[np.random.randint(0, mat.shape[0], size=1)] for centroid_index in range(self.num_centroid)} 
            
            centroid_array = np.vstack([val for val in self.centroids.values()]) # converting the centroids dictionary to an array containing the k-centroid values
            # print(centroid_array)
            distances = cdist(mat, centroid_array) # will return a list of distances between the point of interest in the dataset and all the centroids
            # print("Distances:", distances)

            # find the closest centroid for the current point
            closest_centroid = np.argmin(distances, axis=1)

            # saving the 1D matrix of the cluster designation into the "closest_centroid_mat" attribute to access in the predict function
            self.closest_centroid_mat = closest_centroid
            # print("Closest Centroid:", closest_centroid)

            # update the new_centroids to be the mean of all the points assigned to it
            for i in range(self.num_centroid):
                self.new_centroids[i] = np.mean(mat[closest_centroid == i], axis=0)
                new_centroid_array = np.vstack([val for val in self.new_centroids.values()]) # converting the new centroids dictionary to an array containing the k-centroid values
                print("NEW:", new_centroid_array)
                # print("New Centroids:", self.centroids)
            
            # calculating difference between m_i and m_{i-1}
            difference = np.sum(centroid_array - new_centroid_array) 

            # if the diff between m_i and m_{i-1} is less than tolerance given, save m_i as centroids and break from loop
            if difference < self.tolerance:
                print(_)
                # print("diff < tolerance")
                self.centroids = self.new_centroids
                break

            # if it did not "converge", then repeat by replacing m_i as the old centroid to compute m_{i+1}
            self.centroids = self.new_centroids

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        # Making sure mat is of the correct type
        if type(mat) != np.ndarray:
            raise TypeError("mat must be a numpy array")
        
        # make sure the fit function has been called before running the predict function
        if self.closest_centroid_mat is None:
            raise AttributeError("Call 'fit' before calling 'predict'")

        # making sure the # of features in mat is the same as the # of features in the centroids
        if mat.shape[1] != len(self.centroids[0]):
            raise ValueError("The number of features in mat must be the same as the number of features in the centroids")
        
        # making sure the matrix that the fit function was called on is the same as the matrix provided in the predict function
        if (self.mat != mat).all():
            raise ValueError("The matrix provided must be the same as the matrix used in the fit function")

        return self.closest_centroid_mat

        # return cluster_label

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        if self.mat is None:
            raise AttributeError("Call 'fit' before calling 'get_error'")
        
        self.error = None
        centroid_array = np.vstack([val for val in self.centroids.values()]) # converting the centroids dictionary to an array containing the k-centroid values

        # calculate the distance between the points in the dataset and the centroids
        # using sqeuclidean distance metric to compute the squared euclidean distance between the two points
        distances = cdist(self.mat, centroid_array)
        sum_error = 0

        for point_ix in range(len(self.mat)):
            # identify which centroid the point belongs to
            cluster_ix = self.closest_centroid_mat[point_ix]
            # calculate the squared-mean error of the fit model
            error = np.sum(distances[point_ix, cluster_ix]**2)
            sum_error += error

        self.error = sum_error
        return self.error


    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        # converting the centroids dictionary to a 'k x m' matrix containing cluster centroids
        centroid_array = np.vstack([val for val in self.centroids.values()])

        
        return centroid_array



mat, labels = make_clusters()
mat2, labels2 = make_clusters(seed=30)
# print(mat)
K = KMeans(k = 3)
print(K.fit(mat=mat))
# print(K.get_centroids())
print(K.predict(mat))
# print(mat)
print(K.get_error())

val = cdist([[0, 0], [1, 1]], [[1, 1], [2, 2]], 'sqeuclidean')
print(val)