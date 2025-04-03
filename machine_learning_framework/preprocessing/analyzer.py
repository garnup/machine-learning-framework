import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from machine_learning_framework.preprocessing import preprocessing_utils as pre_utils

class Analyzer():
    def __init__(self, df):
        self.df = df

    def describe(self):
        return self.df.describe()

    def drop_missing_data(self):
        self.df = self.df.dropna(axis=0)

    def drop_columns(self, cols):
        self.df = self.df.drop(cols, axis=1)

    def _get_numerical_and_categorical_features(self):
        all_cols = self.df.columns
        numerical_cols = self.df._get_numeric_data().columns
        categorical_cols = list(set(all_cols) - set(numerical_cols))
        return numerical_cols, categorical_cols

    def _encodeOneHot(self, col):
        self.df = pd.concat([self.df, pd.get_dummies(self.df[col], dtype=float)], axis=1)
        self.df = self.df.drop(col, axis=1)
        return self.df

    def encode_one_hot(self, cols):
        for col in cols:
            self._encodeOneHot(col)
        return self.df

    ''' Encode features in the dataframe we have stored. This method will use standard scaling
    on all the numerical features and, by default, one-hot encoding for any identified
    categorical features. If you want to do label encoding, it's recommended that you call
    encode_label separately
    TODO: this method isn't being used, and hasn't been tested
    '''
    def encode_features(self, cols=[]):
        if cols == None:
            cols = []
        all_numerical_cols, all_categorical_cols = self._get_numerical_and_categorical_features()
        numerical_cols_to_encode = list(set(all_numerical_cols).intersection(set(cols)))
        categorical_cols_to_encode = list(set(all_categorical_cols).intersection(set(cols)))
        for col in categorical_cols_to_encode:
            self._encodeOneHot(col)
        scaler = StandardScaler()
        self.df[numerical_cols_to_encode] = scaler.fit_transform(self.df[numerical_cols_to_encode])

    def encode_label(self, cols):
        for col in cols:
            self.df[col], _ = pre_utils.integer_encode(self.df[col])
        return self.df

    def encode_standard(self, cols):
        encoded_df = pre_utils.normalize_data(self.df[cols])
        self.df[cols] = encoded_df
        return self.df

    ''' Randomly shuffle all the rows in the dataframe (and reset the index)
    '''
    def shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        return self.df

    def retrieve_data(self) -> pd.DataFrame:
        return self.df

    def plot_correlationMatrix(self, cols=None):
        df = self.df if cols==None else self.df[cols]
        sns.heatmap(df.corr())

    def plot_pairPlot(self, cols=None):
        df = self.df if cols==None else self.df[cols]
        sns.pairplot(df)

    def plot_histograms_numeric(self, cols=None, bins=None):
        if cols == None:
            cols, _ = self._get_numerical_and_categorical_features()
        sns.histplot(cols, bins=bins)

    def plot_histograms_categorical(self, cols=None):
        if cols == None:
            _, cols = self._get_numerical_and_categorical_features()
        sns.histplot(cols)
    
    def plot_boxPlot(self, cols=None):
        if cols == None:
            cols, _ = self._get_numerical_and_categorical_features()
        sns.boxplot(cols)