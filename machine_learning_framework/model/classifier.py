import logging
import warnings
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

Model = namedtuple("Model", "name model_family model_type predict_type")

logger = logging.getLogger(name="Classification module")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import lightgbm as lgb


class Classifier(ABC):
    def __init__(
        self,
        name: str,
        model_family: str,
        random_state: int,
        n_jobs=1,
        **model_configs,
    ):
        """
        Creates a classification object
        Parameters
        ----------
        random_state : seed used when performing any random operation
        name : classification model name
        model_family : family of the classification model (e.g. neural network, trees)
        n_jobs : number of parallel jobs that the model can use
        """
        self.random_state = random_state
        self.model_details = Model(
            name=name,
            model_family=model_family,
            model_type="classification",
            predict_type="classification",
        )
        self.model_additional_configs = model_configs
        self.n_jobs = n_jobs

        self.model = None
        self.fitted_model = None
        self.training_set_true = None
        self.training_set_pred = None
        self.grid_search_enabled = False
        self.model_params = {}

    def get_classification_report(self) -> dict:
        """
        Return classification report
        Returns
        -------
        dict of classification metrics
        """
        from sklearn.metrics import classification_report
        return classification_report(self.training_set_true, self.training_set_pred, output_dict=True)

    def get_performance_metrics(self, metrics_list: List[str]):
        """
        Computes performance metrics
        Parameters
        ----------
        metrics_list : list of metrics to compute
        Returns
        -------
        dict of computed metrics
        """
        metrics_dict = {}
        for metric in metrics_list:
            # Assuming metrics functions are defined, similar to the regression example
            metrics_dict[metric] = metrics_processes_dict[metric](
                y_true=self.training_set_true, y_pred=self.training_set_pred
            )

        return metrics_dict

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

    def fit(self, x_data: np.ndarray, y_data: np.ndarray, **fit_configs: dict) -> Model:
        """
        Fits a machine learning model and creates a fitted model
        Parameters
        ----------
        x_data :  numpy array of input X_data for the model fitting
        y_data : numpy array of target data
        fit_configs : dict of additional parameters for fitting the model
        Returns
        -------
        return a fitted model
        """
        self.fitted_model = self.model.fit(x_data, y_data, **fit_configs)
        self.training_set_true = y_data
        self.training_set_pred = self.model.predict(x_data)
        return self.fitted_model

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        """
        Predicts using the fitted model
        Parameters
        ----------
        x_data : numpy array of input data
        Returns
        -------
        numpy array of predictions
        """
        res_array = self.fitted_model.predict(x_data)
        return res_array

    def optimize_hyperparameters(
        self,
        x_data,
        y_data,
        params_dict,
        sample=False,
        cv_count=3,
        score_function="accuracy",
    ):
        """
        Optimizes hyperparameters using GridSearchCV
        Parameters
        ----------
        x_data : numpy array of input data
        y_data : numpy array of target data
        params_dict : dictionary of parameters to optimize
        sample : whether to sample a portion of the data
        cv_count : number of cross-validation folds
        score_function : scoring function to use
        Returns
        -------
        GridSearchCV object and best parameters
        """
        if sample is True:
            selected_indices = random.sample(
                range(0, len(x_data)), int(len(x_data) * 0.2)
            )
            x_data_final = x_data[selected_indices]
            y_data_final = y_data[selected_indices]
        else:
            x_data_final = x_data
            y_data_final = y_data

        gs = GridSearchCV(self.model, params_dict, cv=cv_count, scoring=score_function)
        gs.fit(x_data_final, y_data_final)

        print(f"\nBest Params --> {gs.best_params_}")

        return gs, gs.best_params_

    @abstractmethod
    def create_object_model(self):
        """
        Create the classification model object
        """

    def update_model_params(self, new_params: dict) -> dict:
        """
        Update the model parameters
        """
        for param in new_params:
            self.model_params[param] = new_params[param]
        self.create_object_model()


class LogisticRegressionClassifier(Classifier):
    def __init__(self, random_state: int = 1, n_jobs: int = 1):
        """
        Create a logistic regression model
        Parameters
        ----------
        random_state : seed for random model
        """
        super().__init__(
            name="Logistic_Regression",
            model_family="logistic_regression",
            random_state=random_state,
            n_jobs=n_jobs,
        )

        self.create_object_model()

    def create_object_model(self):
        self.model = LogisticRegression(random_state=self.random_state, n_jobs=self.n_jobs)


class RandomForestClassification(Classifier):
    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 1,
        max_depth: int = None,
        n_jobs: int = 1,
    ):
        """
        Create a random forest classification model
        Parameters
        ----------
        n_estimators : number of trees in the forest
        random_state : seed for random model
        max_depth : maximum depth of the tree
        """
        super().__init__(
            name="Random_Forest",
            model_family="tree_classifier",
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.model_params["n_estimators"] = n_estimators
        self.model_params["max_depth"] = max_depth
        self.create_object_model()

    def create_object_model(self):
        self.model = RandomForestClassifier(
            n_estimators=self.model_params["n_estimators"],
            max_depth=self.model_params["max_depth"],
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )


class SVMClassifier(Classifier):
    def __init__(self, kernel: str = 'rbf', random_state: int = 1):
        """
        Create a Support Vector Machine (SVM) classification model
        Parameters
        ----------
        kernel : kernel type to be used in the algorithm
        random_state : seed for random model
        """
        super().__init__(
            name="SVM_Classifier",
            model_family="svm",
            random_state=random_state,
        )
        self.model_params["kernel"] = kernel
        self.create_object_model()

    def create_object_model(self):
        self.model = SVC(kernel=self.model_params["kernel"], random_state=self.random_state)


class NNClassifier(Classifier):
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=200,
        random_state=1,
        n_jobs: int = 1,
    ):
        """
        Create a neural network classification model
        Parameters
        ----------
        hidden_layer_sizes : tuple, length = n_layers - 2, default (100,)
        learning_rate : learning rate schedule
        learning_rate_init : initial learning rate
        max_iter : maximum number of iterations
        random_state : seed for random model
        """
        super().__init__(
            name="Neural_Network_Classifier",
            model_family="neural_network",
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.model_params["hidden_layer_sizes"] = hidden_layer_sizes
        self.model_params["learning_rate"] = learning_rate
        self.model_params["learning_rate_init"] = learning_rate_init
        self.model_params["max_iter"] = max_iter
        self.create_object_model()

    def create_object_model(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=self.model_params["hidden_layer_sizes"],
            learning_rate=self.model_params["learning_rate"],
            learning_rate_init=self.model_params["learning_rate_init"],
            max_iter=self.model_params["max_iter"],
            random_state=self.random_state,
        )


class LightGBMClassifier(Classifier):
    def __init__(
        self,
        objective="binary",
        metric=None,
        boosting_type="gbdt",
        learning_rate=0.1,
        num_leaves=31,
        n_estimators=100,
        random_state: int = 1,
        n_jobs: int = 1,
    ):
        """
        Create a LightGBM classification model
        Parameters
        ----------
        objective : objective function
        metric : evaluation metric(s)
        boosting_type : boosting type
        learning_rate : learning rate
        num_leaves : number of leaves in one tree
        n_estimators : number of boosting rounds
        random_state : seed for random model
        """
        super().__init__(
            name="LightGBM_Classifier",
            model_family="tree_classifier",
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.model_params["objective"] = objective
        self.model_params["metric"] = metric if metric is not None else ["binary_logloss", "binary_error"]
        self.model_params["boosting_type"] = boosting_type
        self.model_params["learning_rate"] = learning_rate
        self.model_params["num_leaves"] = num_leaves
        self.model_params["n_estimators"] = n_estimators
        self.create_object_model()

    def create_object_model(self):
        self.model = lgb.LGBMClassifier(
            objective=self.model_params["objective"],
            metric=self.model_params["metric"],
            boosting_type=self.model_params["boosting_type"],
            learning_rate=self.model_params["learning_rate"],
            num_leaves=self.model_params["num_leaves"],
            n_estimators=self.model_params["n_estimators"],
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
