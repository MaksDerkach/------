import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer



class DatasetRelable:
    def __init__(self, dataset: pd.DataFrame,
                       target_col: str,
                       time_col: str=None,
                       split_edges: list=[0.3, 0.533, 0.766],
                       data_mode: str='already_processed',
                       cat_cols: list=None,
                       numeric_cols: list=None,
                       ordinal_cols: list=None) -> None:
        """
        `dataset`: source data for processing
        `target_col`: target column

        `data_mode`: Two possible values:

            - 'already_processed'

            - 'source_data'

        `split_edges`: List of float to split data into 4 subsets: 
            [D_init, D_train, D_control_1, D_control_2]
        """

        # dataset info
        self.dataset = dataset
        self.target_col = target_col
        self.data_mode = data_mode

        self._columns = dataset.columns
        self._service_columns = [target_col, time_col, 'split_group', 'target_score']

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
        """
        Showing basic statistics of dataset: `shape` and `disbalance`
        """
        for group in self._edges_mapping:
            print(self._get_statistic_per_group(group))


    def _get_statistic_per_group(self, group):
        group_shape = self.dataset[self.dataset.split_group == self._edges_mapping[group]].shape
        fraud_cnt = self.dataset[
            (self.dataset.split_group == self._edges_mapping[group])
            & (self.dataset[self.target_col] == 1)
            ].shape[0]
        
        disbalance = fraud_cnt / group_shape[0] * 100

        return f"{self._edges_mapping[group]}: shape is {group_shape}\nDisbalance: {disbalance:.1f}% \n"
    

    def _prepare_data_for_train(self, split_group):
        X = (
            self.dataset[self.dataset.split_group == split_group]
            .drop(columns=self._service_columns, axis=1, errors='ignore')
            )
        y = self.dataset[self.dataset.split_group == split_group][self.target_col]

        return X, y
    

    def _prepare_index_for_train(self, split_group):
        train_indexes = (
            self.dataset[self.dataset.split_group == split_group]
            .index.to_numpy()
        )
        train_columns = self.dataset.drop(columns=self._service_columns, axis=1, errors='ignore').columns

        return train_indexes, train_columns



    def train_relabler(self, parameters: dict,
                             model,
                             scoring: str='roc_auc'):
        """
        Train model-relabler on 'D_init' part of dataset
        """

        inds, cols = self._prepare_index_for_train('D_init')

        M_rl = GridSearchCV(estimator=model,
                            param_grid=parameters,
                            refit=True,
                            scoring=scoring,
                            cv=3
                           ).fit(self.dataset.loc[inds, cols],
                                 self.dataset.loc[inds, self.target_col])
        
        self.relabler = M_rl
        self.dataset.loc[inds, 'target_score'] = M_rl.predict_proba(self.dataset.loc[inds, cols])[:, 1]

        print(f'Scoring "{scoring}" is {M_rl.best_score_}')
        print(f'Best params: {M_rl.best_params_}')


    def plot_relabler_distribution(self):
        """
        Plot distribution of D_init data by target
        """
        data_for_plot = self.dataset[[self.target_col, 'target_score']]

        sns.displot(data_for_plot, x='target_score',
                    hue=self.target_col, stat='desnsity', common_norm=False)





    def train_base_model(self, parameters: dict,
                               model,
                               scoring: str='roc_auc'):
        """
        Train baseline model on `D_train` part of dataset without relabling
        then validate on `D_control_1`
        
        """

        train_ind, cols = self._prepare_data_for_train('D_train')
        val_ind, _ = self._prepare_data_for_train('D_control_1')

        M_base = GridSearchCV(estimator=model,
                            param_grid=parameters,
                            refit=True,
                            scoring=scoring,
                            cv=3
                           ).fit(self.dataset.loc[train_ind, cols],
                                 self.dataset.loc[train_ind, self.target_col])
        
        self.base_model = M_base

        print(f'Scoring "{scoring}" is {M_base.best_score_}')
        print(f'Best params: {M_base.best_params_}')
    


    def train_base_relable_model(self):
        """
        """
        train_ind, cols = self._prepare_data_for_train('D_train')



    def compare_models(self):
        pass



    def _change_bounds(self):
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
