import numpy as np
import matplotlib.pyplot as plt

def make_clusters(
        n: int = 500, 
        m: int = 2, 
        k: int = 3, 
        bounds: tuple = (-10, 10),
        scale: float = 1,
        seed: int = 42) -> (np.ndarray, np.ndarray):
    """
    creates some clustered data

    inputs:
        n: int
            number of observations
        m: int
            number of features
        k: int
            number of clusters
        bounds: tuple
            minimum and maximum bounds for cluster grid
        scale: float
            standard deviation of normal distribution
        seed: int
            random seed

    outputs:
        (np.ndarray, np.ndarray)
            returns a 2D matrix of `n` observations and `m` features that are clustered into `k` groups
            returns a 1D array of `n` size that defines the cluster origin for each observation
    """
    np.random.seed(seed)
    assert k <= n

    labels = np.sort(np.random.randint(0, k, size=n))
    centers = np.random.uniform(bounds[0], bounds[1], size=(k,m))
    mat = np.vstack([
        np.random.normal(
            loc=centers[idx], 
            scale=scale, 
            size=(np.sum(labels==idx), m))
        for idx in np.arange(0, k)])

    return mat, labels


def plot_clusters(
        mat: np.ndarray, 
        labels: np.ndarray, 
        filename: str =None):
    """
    inputs:
        mat: np.ndarray
            a 2D matrix where each row is an observation and each column is a feature
        labels: np.ndarray
            a 1D array where each value represents an integer cluster that an observation belongs to
        filename: str
            an optional value to save a figure to a file
    """

    plt.figure(figsize=(5,5), dpi=200)
    plt.scatter(
        mat[:,0], 
        mat[:,1], 
        c=labels)

    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

def plot_with_number(
        mat: np.ndarray, 
        labels: np.ndarray,
        filename: str =None
    ):
    plt.figure(figsize=(5,5), dpi=200)
    N = len(mat)
    plt.scatter(mat[:,0], mat[:,1], c=labels)
    bad_score = [169, 170, 171, 173, 174, 177, 179, 180, 188, 193, 197, 202, 203, 205, 206, 207, 211, 216, 217, 218, 221, 226, 227, 231, 232, 235, 239, 242, 243, 244, 247, 248, 253, 255, 257, 258, 259, 260, 265, 269, 271, 273, 276, 277, 278, 279, 282, 285, 288, 292, 298, 300, 301, 308, 309, 310, 311, 316, 319, 327, 329, 334, 339, 344, 345, 348, 349, 350, 351, 359, 365, 370, 371, 374, 377, 378, 379, 380, 390, 394, 395, 396, 397, 399, 400, 403, 404, 406, 407, 409, 410, 415, 417, 420, 421, 424, 431, 432, 434, 438, 439, 441, 443, 448, 449, 450, 454, 455, 456, 465, 467, 470, 472, 476, 477, 479, 480, 490, 494, 496, 497, 499]
    for i in range(mat.shape[0]):
        if i in bad_score:
            plt.text(mat[i,0], mat[i,1], "x", fontsize=3, color='red')


    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

def plot_multipanel(
        mat: np.ndarray,
        truth: np.ndarray,
        pred: np.ndarray,
        score: np.ndarray,
        filename: str = None):
    """
    Plots a multipanel figure visualizing the efficiency of truth, prediction, 
    and silhouette scoring on a provided dataset

    inputs:
        mat: np.ndarray
            a 2D matrix where each row is an observation and each column is a feature
        truth: np.ndarray
            a 1D array where each value represents a true integer cluster that an observation belongs to
        pred: np.ndarray
            a 1D array where each value represents a predicted integer cluster than an observation belongs to
        score: np.ndarray
            a 1D array where each value represents a float for the silhouette score of that observation
        filename: str
            an optional value to save a figure to a file
    """

    fig, axs = plt.subplots(1, 3, figsize=(9,3), dpi=200)
    
    cvars = [truth, pred, score]
    names = ["True Cluster Labels", "Predicted Cluster Labels", "Silhouette Scores"]
    cmaps = [None, None, "seismic"]
    for idx, ax in enumerate(axs):
        sub = ax.scatter(
            mat[:,0],
            mat[:,1],
            c=cvars[idx],
            cmap=cmaps[idx])
        ax.set_title(names[idx])
        if idx == 2:
            plt.colorbar(sub, ax=ax)
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

# mat, labels = make_clusters()
# plot_with_number(mat, labels, "plot_with_neg_silhouette_score_marked.png")