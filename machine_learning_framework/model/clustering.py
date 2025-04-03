import logging
import random
import warnings
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, GridSearchCV

Model = namedtuple("Model", "name model_family model_type predict_type")

logger = logging.getLogger(name="Clustering module")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


class Clustering(ABC):
    def __init__(
        self,
        name: str,
        model_family: str,
        random_state: int,
        n_jobs=1,
        **model_configs,
    ):
        """
        Creates a clustering object
        Parameters
        ----------
        random_state : seed used when performing any random operation
        name : clustering model name
        model_family : family of the clustering model (e.g. kmeans, hierarchical)
        n_jobs : number of parallel jobs that the model can use
        """
        self.random_state = random_state
        self.model_details = Model(
            name=name,
            model_family=model_family,
            model_type="clustering",
            predict_type="unsupervised",
        )
        self.model_additional_configs = model_configs
        self.n_jobs = n_jobs

        self.model = None
        self.fitted_model = None
        self.training_set = None
        self.cluster_labels = None
        self.grid_search_enabled = False
        self.model_params = {}

    @abstractmethod
    def get_cluster_centers(self):
        """
        Return cluster centers
        Returns
        -------
        cluster centers
        """

    @abstractmethod
    def get_cluster_labels(self):
        """
        Return cluster labels
        Returns
        -------
        cluster labels
        """

    @classmethod
    def get_train_test_data(
        cls,
        input_data: pd.DataFrame,
        train_percentage: float = 0.75,
        random_state: int = 1,
    ):
        """
        Split the input data into train and test parts
        Parameters
        ----------
        input_data : input_dataframe to split
        train_percentage : percentage to use for train
        random_state : seed to use for running the model

        Returns
        -------
        training and test dataframes
        """
        return train_test_split(
            input_data, train_size=train_percentage, random_state=random_state
        )

    def fit(self, x_data: np.ndarray, **fit_configs: dict) -> Model:
        """
        Fits a clustering model and creates a fitted model
        Parameters
        ----------
        x_data :  numpy array of input X_data for the model fitting
        fit_configs : dict of additional parameters for fitting the model
        Returns
        -------
        return a fitted model
        """
        self.fitted_model = self.model.fit(x_data, **fit_configs)
        self.training_set = x_data
        self.cluster_labels = self.model.labels_
        return self.fitted_model

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        res_array = self.model.predict(x_data)
        return res_array

    def optimize_hyperparameters(
        self,
        x_data,
        params_dict,
        sample=True,
        cv_count=3,
        score_function="adjusted_rand_score",
    ):
        if sample is True:
            selected_indices = random.sample(
                range(0, len(x_data)), int(len(x_data) * 0.2)
            )
            x_data_final = x_data[selected_indices]
        else:
            x_data_final = x_data

        gs = GridSearchCV(self.model, params_dict, cv=cv_count, scoring=score_function)
        gs.fit(x_data_final)

        print(f"\nBest Params --> {gs.best_params_}")

        return gs, gs.best_params_

    @abstractmethod
    def create_object_model(self):
        """
        Create the clustering model object
        """

    def update_model_params(self, new_params: dict) -> dict:
        """
        Update the model parameters
        """
        for param in new_params:
            self.model_params[param] = new_params[param]
        self.create_object_model()

    def get_silhouette_score(self):
        from machine_learning_framework.postprocessing.metrics import get_silhouette_score
        return get_silhouette_score(self.training_set, self.cluster_labels, self.random_state)


class KMeansClustering(Clustering):
    def __init__(self, n_clusters: int = 8, random_state: int = 1, n_jobs: int = 1):
        """
        Create a k-means clustering model
        Parameters
        ----------
        n_clusters : number of clusters
        random_state : seed for random model
        """
        super().__init__(
            name="KMeans_Clustering",
            model_family="kmeans",
            random_state=random_state,
            n_jobs=n_jobs,
        )

        self.model_params["n_clusters"] = n_clusters
        self.create_object_model()

    def create_object_model(self):
        self.model = KMeans(
            n_clusters=self.model_params["n_clusters"],
            random_state=self.random_state,
        )

    def get_cluster_centers(self):
        return self.model.cluster_centers_

    def get_cluster_labels(self):
        return self.model.labels_


class AgglomerativeClusteringModel(Clustering):
    def __init__(self, n_clusters: int = 8, random_state: int = 1):
        """
        Create an agglomerative clustering model
        Parameters
        ----------
        n_clusters : number of clusters
        random_state : seed for random model
        """
        super().__init__(
            name="Agglomerative_Clustering",
            model_family="hierarchical",
            random_state=random_state,
        )

        self.model_params["n_clusters"] = n_clusters
        self.create_object_model()

    def create_object_model(self):
        self.model = AgglomerativeClustering(
            n_clusters=self.model_params["n_clusters"]
        )

    def get_cluster_centers(self):
        # Agglomerative Clustering does not have cluster centers
        return None

    def get_cluster_labels(self):
        return self.model.labels_


class DBSCANClustering(Clustering):
    def __init__(self, eps: float = 0.5, min_samples: int = 5, random_state: int = 1):
        """
        Create a DBSCAN clustering model
        Parameters
        ----------
        eps : maximum distance between two samples for them to be considered as in the same neighborhood
        min_samples : number of samples (or total weight) in a neighborhood for a point to be considered as a core point
        random_state : seed for random model
        """
        super().__init__(
            name="DBSCAN_Clustering",
            model_family="density_based",
            random_state=random_state,
        )

        self.model_params["eps"] = eps
        self.model_params["min_samples"] = min_samples
        self.create_object_model()

    def create_object_model(self):
        self.model = DBSCAN(
            eps=self.model_params["eps"],
            min_samples=self.model_params["min_samples"],
        )

    def get_cluster_centers(self):
        # DBSCAN does not have cluster centers
        return None

    def get_cluster_labels(self):
        return self.model.labels_


class GaussianMixtureClustering(Clustering):
    def __init__(self, n_components: int = 1, random_state: int = 1, n_jobs: int = 1):
        """
        Create a Gaussian Mixture clustering model
        Parameters
        ----------
        n_components : number of mixture components
        random_state : seed for random model
        """
        super().__init__(
            name="GaussianMixture_Clustering",
            model_family="mixture_model",
            random_state=random_state,
            n_jobs=n_jobs,
        )

        self.model_params["n_components"] = n_components
        self.create_object_model()

    def create_object_model(self):
        self.model = GaussianMixture(
            n_components=self.model_params["n_components"],
            random_state=self.random_state,
        )

    def get_cluster_centers(self):
        return self.model.means_

    def get_cluster_labels(self):
        return self.model.predict(self.training_set)
