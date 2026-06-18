from utils import *
import numpy as np
import scipy as scipy


def training_data_eigenvectors():
    """
    This function saves the eigenvectors computed from the training data
    These eigenvectors will be used for evaluating the noise_test and mask_test

    Calculate the mean and s.d. of each column
    Standardised the images, adding 1e-10 to avoid diving by 0
    Center the standardised images

    Calculate the covaraince matrix and the eigenvectors
    and eigenvalues.
    Sort the eigenvalues in descending order and reorder
    based on importance from largest to smallest

    Align the eigenvectors with the eigenvalues, select
    the top 63 eigenvectors and return them.

    """
    images,training_labels = get_dataset('train')
    feature_means = np.mean(images, axis=0) 
    feature_stds = np.std(images, axis=0) 
    standardised_images = (images - feature_means) / (feature_stds + 1e-10) 
    images_centered = standardised_images - np.mean(standardised_images, axis=0)

    covarianceMatrix = np.cov(images_centered, rowvar=False)  
    eigenvalues, eigenvectors = np.linalg.eigh(covarianceMatrix) 
    sorted_values = np.argsort(eigenvalues)[::-1] 
    eigenvalues = eigenvalues[sorted_values] 
    eigenvectors = eigenvectors[:, sorted_values] 
    selected_eigenvectors = eigenvectors[:,:63] 

    return selected_eigenvectors 



def image_to_reduced_feature(images, split='train'):
    """
    Function to reduce the dimensionality of the inputted dataset

    Calculate the mean and s.d. of each column
    Standardised the images, adding 1e-10 to avoid diving by 0
    Center the standardised images

    If the split is None,meaning if the split is noise
    or mask, then return the saved eigenvectors from the
    training data and reduce the data with those eigenvectors

    If the split is train, then use the saved eigenvectors
    and reduce the training dataset using those eigenvectors

    """
    images_means = np.mean(images, axis=0) 
    images_stds = np.std(images, axis=0) 
    standardised_images = (images - images_means) / (images_stds + 1e-10) 
    images_centered = standardised_images - np.mean(standardised_images, axis=0) 

    if split is None: 
        saved_eigenvectors = training_data_eigenvectors() 
        pcatrain_data_2 = np.dot(images_centered,saved_eigenvectors) 
        return pcatrain_data_2
    else:
        saved_eigenvectors_training = training_data_eigenvectors() 
        pcatrain_data = np.dot(images_centered, saved_eigenvectors_training) 
    return pcatrain_data


def training_model(train_features, train_labels):
    """
    Create a knn model with k and weight paramemters
    Use the fit function to store the training data and labels
    """
    knn_model = KNN(k=7, weight=0.59)
    knn_model.fit(train_features, train_labels)
    return knn_model


# Intialise the knn model
class KNN:  

    #k number of neighbors
    #Weight for euclidean and cosine distance
    def __init__(self, k=7, weight=0.59): 
        self.k = k 
        self.train_data = None
        self.train_labels = None
        self.weight = weight 

    #Store the training data and labels used in predict
    def fit(self, train_data, train_labels): 
        self.train_data = train_data
        self.train_labels = train_labels

    # calculate distances between test and training data
    def predict(self, test_data):
        distances = self._calculate_distances(test_data)
        return self._calculate_labels(distances) 


    def _calculate_distances(self, test_data):
        """
        Calculate dot product between test and train data
        Normalise the test and training data

        Compute the cosine similarity, adding 1e-10 to avoid dividing by 0
        Calcualting euclidean distance using by expanding the test and training
        data so that broadcasting works. Use the squared differnece to
        find the euclidean distance

        Normalise the calcualted distances and use the weight value
        to compute the combined distance
        """
        dot_product = np.dot(test_data, self.train_data.T) 
        test_data_norms = np.linalg.norm(test_data, axis=1)[:, np.newaxis] 
        train_data_norms = np.linalg.norm(self.train_data, axis=1) 

        cosine_sim = dot_product / (np.clip(test_data_norms * train_data_norms, 1e-10, None)) 
        cosine_dis = 1 - cosine_sim #find the cosine distance


        test_data_reshaped = test_data[:, np.newaxis, :]  
        train_data_reshaped= self.train_data[np.newaxis, :, :]  
        sqrd_diffs = (test_data_reshaped - train_data_reshaped) ** 2  
        euclidean_dis = np.sqrt(np.sum(sqrd_diffs, axis=2)) 


  
        cosine_dis /= np.max(cosine_dis, axis=1, keepdims=True)
        euclidean_dis /= np.max(euclidean_dis, axis=1, keepdims=True)

 
        cosine_eucli_dis = self.weight * cosine_dis + (1 - self.weight) * euclidean_dis
        return cosine_eucli_dis


    def _calculate_labels(self, distances):
        """
        Find distances with the smallest value of k and 
        extract the labels of the nearest neighbor

        Find the most frequent label by counting the 
        occurence of each label and finding the
        label with the most occurences
        """
        sort_distances = np.argsort(distances, axis=1)[:, :self.k] 
        nearest_labels = self.train_labels[sort_distances] 

 
        predicted_labels = np.array([
            np.bincount(nearest_labels[i]).argmax() 
            for i in range(nearest_labels.shape[0]) 
        ])
        return predicted_labels
