import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, f1_score

from joblib import dump, load

from models import Preprocessor, DummyModel

import os
import shutil


class DatasetRelable:
    def __init__(self, dataset: pd.DataFrame,
                       target_col: str,
                       time_col: str=None,
                       time_col_type='int',
                       preprocessed: bool=True) -> None:
        """
        `dataset`: source data for processing
        `target_col`: target column
        `time_col`: along this column the dataset will be devided into subgroups

        `time_col_type`: data type to plot the distribution, only 2 possible values:
            - 'int' means that time_col can`t be converted to pd.datetime
            - 'datetime' means that time_col can be converted to pd.datetime

        `preprocessed`: Three possible values:
            - 'True' means that categorical and numeric features already processed
            - 'False' means raw data that need to be processed
        """

        # dataset info
        self.dataset = dataset

        if time_col is not None:
            self.dataset = self.dataset.sort_values(time_col, ascending=True).reset_index(drop=True)

        self.target_col = target_col
        self._is_preprocessed = preprocessed
        self.time_col = time_col
        self.time_col_type = time_col_type

        # service attributes
        self._edges_mapping = {0: 'D_init', 1: 'D_train',
                               2: 'D_control_1', 3: 'D_control_2'}
        
        self._split_col = 'split_group'
        self._relable_score_col = 'target_score'
        self._relable_target_col = 'target_relabled'
        self._service_columns = [self.target_col, self.time_col, self._split_col,
                                 self._relable_score_col, self._relable_target_col]

        self._models_path = 'models/'
        self._model_relabler_path = self._models_path + 'relabler'
        self._model_base_path = self._models_path + 'base_model'
        self._model_relabled_path = self._models_path + 'relabled_model'

        if not os.path.exists(self._models_path):
            os.makedirs(self._models_path)
        else:
            shutil.rmtree(self._models_path)
            os.makedirs(self._models_path)


        if not self._is_preprocessed:
            self.preprocessor = Preprocessor(self.dataset.drop(columns=self._service_columns, errors='ignore'))
        


    def _get_index_edges(self, edges):
        edges = (np.array(edges) * self.dataset.shape[0]).astype(int)
        return edges
    

    def _split_to_edges(self):
        """
        Split dataset into 4 subsets: 
            [D_init, D_train, D_control_1, D_control_2]
        """
        for n_group, sub_index in enumerate(np.split(self.dataset.index.to_numpy(), self.index_edges)):
            self.dataset.loc[sub_index, self._split_col] = self._edges_mapping[n_group]
    

    def split_data_and_get_stat(self, split_edges: list=[0.3, 0.533, 0.766], n_ticks=10):
        """
        Plot the graph (if `time_col` is not empty) of splitted dataset 
        and show basic statistics of subsets: `shape` and `disbalance`
        """
        # split for 4 subsets
        self.raw_edges = split_edges
        self.index_edges = self._get_index_edges(self.raw_edges)
        self._split_to_edges()

        # show statistic per subset
        self._get_split_statistics()

        if self.time_col is not None:
            if self.time_col_type == 'int':
                sns.histplot(self.dataset, x=self.time_col,
                             bins=self.dataset.shape[0] // 1000,
                             hue=self._split_col)
                
            elif self.time_col_type == 'datetime':
                data_for_plot = self.dataset[[self.time_col, self._split_col, self.target_col]]
                data_for_plot[self.time_col] = pd.to_datetime(data_for_plot[self.time_col]).dt.strftime('%Y-%m-%d %Hh')

                data_for_plot = data_for_plot.groupby([self.time_col, self._split_col]) \
                                             .agg(cnt_events=(self.target_col, 'count')).reset_index() \
                                             .pivot(index=self.time_col, columns=self._split_col, values='cnt_events')
                
                step = data_for_plot.shape[0] // n_ticks
                ticks = range(0, len(data_for_plot.index), step)
                labels = data_for_plot.index[::step]

                data_for_plot.plot.bar(stacked=True, figsize=(14, 4))
                plt.xticks(ticks, labels, rotation=45)

    
    
    def _get_split_statistics(self):
        """
        Showing basic statistics of dataset: `shape` and `disbalance`
        """
        for group in self._edges_mapping:
            print(self._get_statistic_per_group(group))


    def _get_statistic_per_group(self, group):
        group_shape = self.dataset[self.dataset[self._split_col] == self._edges_mapping[group]].shape
        fraud_cnt = self.dataset[
            (self.dataset[self._split_col] == self._edges_mapping[group])
            & (self.dataset[self.target_col] == 1)
            ].shape[0]
        
        disbalance = fraud_cnt / group_shape[0] * 100

        return f"{self._edges_mapping[group]}: shape is {group_shape}\nDisbalance: {disbalance:.1f}% \n"
    

    def _prepare_data_for_train(self, split_group):
        X = (
            self.dataset[self.dataset[self._split_col] == split_group]
            .drop(columns=self._service_columns, errors='ignore')
            )
        y = self.dataset[self.dataset[self._split_col] == split_group][self.target_col]

        return X, y
    

    def _prepare_index_for_train(self, split_group):
        train_indexes = (
            self.dataset[self.dataset[self._split_col] == split_group]
            .index.to_numpy()
        )
        train_columns = self.dataset.drop(columns=self._service_columns, errors='ignore').columns

        return train_indexes, train_columns



    def train_relabler(self, params: dict):
        """
        Train model-relabler on 'D_init' part of dataset
        'params' is a dict-like object with required fields:

            - `model` - is a source model for training
            - `parameters` - grid of parameters for searching best algorithm
            - `scoring` - one or dict of scores
            - `refit`
            - `need_smote`
        """

        inds, cols = self._prepare_index_for_train('D_init')

        X = self.dataset.loc[inds, cols]
        y = self.dataset.loc[inds, self.target_col]

        self.M_relabler = DummyModel(preprocessor=self.preprocessor.pipe, **params).fit(X, y)

        # save current model
        self._save_model(self.M_relabler, self._model_relabler_path, 'model_relabler')



    def plot_relabler_distribution(self, bins_step=0.01, subset='D_init'):
        """
        Plot distribution of `subset` data by target
        """
        bins_range = np.arange(0, 1 + bins_step, bins_step)
        inds, cols = self._prepare_index_for_train(subset)
        
        X = self.dataset.loc[inds, cols]
        y = self.dataset.loc[inds, self.target_col]

        y_proba = self.M_relabler.predict_proba(X)[:, 1]

        data_for_plot = pd.concat([y.reset_index(drop=True), pd.Series(y_proba)], axis=1) \
                          .set_axis([self.target_col, self._relable_score_col], axis=1)
        
        sns.displot(data_for_plot, x=self._relable_score_col,
                    hue=self.target_col, bins=bins_range,
                    stat='density', common_norm=False)




    def train_base_model(self, parameters: dict,
                               model,
                               scoring: str='average_precision',
                               n_jobs=None,
                               verbose=2):
        """
        Train baseline model on `D_train` part of dataset without relabling
        then validate on `D_control_1`
        
        """

        inds, cols = self._prepare_index_for_train('D_train')

        # only for not preprocessed data
        if not self._is_preprocessed:
            model = Pipeline(
                steps=[
                    ('preprocessor', self.preprocessor),
                    ('model', model)
                ]
            )

            parameters = {'model__' + key: values for key, values in parameters.items()}

        M_base = GridSearchCV(estimator=model,
                              param_grid=parameters,
                              refit=True,
                              scoring=scoring,
                              cv=3,
                              verbose=verbose,
                              n_jobs=n_jobs
                            ).fit(self.dataset.loc[inds, cols],
                                  self.dataset.loc[inds, self.target_col])
        
        self.base_model = M_base

        # save current model
        if not os.path.exists(self._model_base_path):
            os.makedirs(self._model_base_path)
        dump(M_base, f"{self._model_base_path}/base_model.joblib")

        print(f'Scoring "{scoring}" is {M_base.best_score_}')
        print(f'Best params: {M_base.best_params_}')
    


    def train_base_relable_model(self, parameters: dict,
                                       model,
                                       TH_legetim: dict,
                                       TH_fraud: dict,
                                       scoring: str='average_precision',
                                       n_jobs=None,
                                       verbose=2):
        """
        `TH_legetim` dict like {start: float, stop: float, step: float}
        `TH_fraud` dict like {start: float, stop: float, step: float}
        """
        inds, cols = self._prepare_index_for_train('D_train')
        val_inds, _ = self._prepare_index_for_train('D_control_1')

        # only for not preprocessed data
        if not self._is_preprocessed:
            model = Pipeline(
                steps=[
                    ('preprocessor', self.preprocessor),
                    ('model', model)
                ]
            )

            parameters = {'model__' + key: values for key, values in parameters.items()}

        self.TH_matrix = []
        model_number = 1
        n_steps = len(np.arange(**TH_legetim)) * len(np.arange(**TH_fraud))
    
        print(f'Eproximate steps: {n_steps}')

        for TH_l in np.arange(**TH_legetim):
            for TH_f in np.arange(**TH_fraud):

                print(f'Step {model_number} ... ')
                self._change_bounds(TH_l, TH_f)

                # Fit model and choose the best hyperparameters
                M_s = GridSearchCV(estimator=model,
                                   param_grid=parameters,
                                   refit=True,
                                   scoring=scoring,
                                   cv=3,
                                   verbose=verbose,
                                   n_jobs=n_jobs
                                   ).fit(self.dataset.loc[inds, cols],
                                         self.dataset.loc[inds, self._relable_target_col])
                
                # save current model
                if not os.path.exists(self._model_relabled_path):
                    os.makedirs(self._model_relabled_path)
                dump(M_s, f"{self._model_relabled_path}/model_relabled_{model_number}.joblib")

                val_score = M_s.score(self.dataset.loc[val_inds, cols], 
                                      self.dataset.loc[val_inds, self.target_col])

                self.TH_matrix.append([f'model_relabled_{model_number}', TH_l, TH_f, val_score])
                model_number += 1
        
        self.TH_matrix_source = pd.DataFrame(self.TH_matrix, columns=['model_name', 'TH_legetim', 'TH_fruad', f'score_{scoring}'])
        self.TH_matrix = self.TH_matrix_source.pivot(index='TH_legetim', columns='TH_fruad', values=f'score_{scoring}')
        

    def _change_bounds(self, TH_l: float, TH_f: float):
        self.dataset[self._relable_target_col] = self.dataset[self.target_col]

        self.dataset.loc[self.dataset[self._relable_score_col] <= TH_l, self._relable_target_col] = 0
        self.dataset.loc[self.dataset[self._relable_score_col] >= TH_f, self._relable_target_col] = 1


    def plot_TH_matrix(self, cmap='BuGn'):
        """
        Plot heatmap based on various thresholds TH_l and TH_f after training model on relabled data `D_train`
        """
        sns.heatmap(self.TH_matrix.sort_index(ascending=False),
                    annot=True, cmap=cmap, fmt='.4g')



    def compare_models(self):
        """
        Compare statistics of M_b and M_s models on `D_control_2` subset
        """

        val_inds, cols = self._prepare_index_for_train('D_control_2')

        model_M_b = self.base_model
        M_b_prediction = model_M_b.predict_proba(self.dataset.loc[val_inds, cols])[:, 1]
        M_b_score = roc_auc_score(self.dataset.loc[val_inds, self.target_col], M_b_prediction)

        return M_b_score
    
    def _save_model(self, object, path, name):
        if not os.path.exists(path):
            os.makedirs(path)
        dump(object, f"{path}/{name}.joblib")
    

    def __str__(self):
        cat_cols, numeric_cols = self.preprocessor.cat_cols, self.preprocessor.numeric_cols

        return f"Data shape is {self.dataset.shape}, target column is '{self.target_col}'\n" +\
            f"Categorical columns: {cat_cols}\n" +\
            f"Numeric columns: {numeric_cols}\n" +\
            f"Time column is '{self.time_col}'\n"

