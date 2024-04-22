import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer



class DatasetRelable:
    def __init__(self, dataset: pd.DataFrame, target_col: str,
                 data_mode: str='already_processed',
                 cat_cols: list=None, numeric_cols: list=None,
                 ordinal_cols: list=None, time_col: str=None,
                 split_edges: list=[0.25, 0.5, 0.75]) -> None:
        
        # dataset info
        self.dataset = dataset
        self.target_col = target_col
        self.data_mode = data_mode
        self._columns = dataset.columns

        # columns parameters
        self.cat_cols = cat_cols
        self.numeric_cols = numeric_cols
        self.time_col = time_col

        self.raw_edges = split_edges
        self.index_edges = self._get_index_edges(self.raw_edges)

        self._edges_mapping = {0: 'D_init', 1: 'D_train',
                               2: 'D_control_1', 3: 'D_control_2'}
        


    def _get_index_edges(self, edges):
        edges = (np.array(edges) * self.dataset.shape[0]).astype(int)
        return edges
    

    def _split_to_edges(self):
        for n_group, sub_index in enumerate(np.split(self.dataset.index.to_numpy(), self.index_edges)):
            self.dataset.loc[sub_index, 'split_group'] = self._edges_mapping[n_group]
    

    def _plot_split_data(self):
        if self.time_col is not None:
            sns.histplot(self.dataset, x=self.time_col,
                         bins=self.dataset.shape[0] // 1000,
                         hue='split_group')
    
    
    def _get_split_statistics(self):
        pass


    def __str__(self):
        return f"Data shape is {self.dataset.shape}, target column is '{self.target_col}'\n" +\
            f"Categorical columns: {self.cat_cols}\n" +\
            f"Numeric columns: {self.numeric_cols}\n" +\
            f"Time column is '{self.time_col}'\n" +\
            f"Split edges: {self.raw_edges}"


    def auto_selection_dtypes(self):
        """
        Automatic selection of data types
        """

        self.cat_cols = []
        self.numeric_cols = []     

        for col_name, dtype in self.dataset.drop(columns=[self.target_col]).dtypes.items():
            if type(dtype) in [np.dtypes.Int64DType, np.dtypes.Float64DType]:
                self.numeric_cols.append(col_name)
            elif type(dtype) in [np.dtypes.ObjectDType]:
                self.cat_cols.append(col_name)


    def preprocess_data(self):
        numeric_transform = Pipeline(
            steps=[('imputer', SimpleImputer(strategy='median')),
                   ('scaler', StandardScaler())]
        )

        categorical_transform = Pipeline(
            steps=[('encoder', OneHotEncoder(handle_unknown='infrequent_if_exist'))]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('numeric', numeric_transform, self.numeric_cols),
                ('categorical', categorical_transform, self.cat_cols)
            ]
        )

        self.preprocess_pipeline = preprocessor
        return preprocessor


    def plot_data(self, mode='whole'):
        if self.time_col is not None:
            if mode == 'whole':
                sns.histplot(self.dataset, x=self.time_col,
                             bins=self.dataset.shape[0] // 1000,
                             hue=self.target_col)
                
            elif mode == 'edges':
                sns.histplot(self.dataset, x=self.time_col,
                             bins=self.dataset.shape[0] // 1000,
                             hue='split_group')
            else:
                raise Exception(f"No such 'mode'='{mode}'")
        else:
            raise Exception("There is no 'time_col'")

