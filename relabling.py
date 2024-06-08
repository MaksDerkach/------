import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from sklearn.metrics import roc_auc_score, average_precision_score, PrecisionRecallDisplay, RocCurveDisplay

from joblib import dump, load

from models import Preprocessor, DummyModel, Splitter

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

        self.target_col = target_col
        self._is_preprocessed = preprocessed
        self.time_col = time_col
        self.time_col_type = time_col_type

        # service attributes
        self._bounds_mapping = {0: 'D_init', 1: 'D_train', 2: 'D_control_1', 3: 'D_control_2'}
        
        self._split_col = 'split_group'
        self._relable_score_col = 'target_score'
        self._relable_target_col = 'target_relabled'
        self._service_columns = [self.target_col, self.time_col, self._split_col,
                                 self._relable_score_col, self._relable_target_col]

        self._models_path = 'models/'
        self._model_relabler_path = self._models_path + 'relabler_model'
        self._model_baseline_path = self._models_path + 'baseline_model'
        self._model_relabled_path = self._models_path + 'relabled_model'

        if not os.path.exists(self._models_path):
            os.makedirs(self._models_path)
        else:
            shutil.rmtree(self._models_path)
            os.makedirs(self._models_path)


        if not self._is_preprocessed:
            self.preprocessor = Preprocessor(self.dataset.drop(columns=self._service_columns, errors='ignore'))
    

    def split_data(self, bounds: list=[0.3, 0.533, 0.766], show_split=False):
        """
        Split the data into 4 groups: D_init, D_train, D_control_1, D_control_2.
        Then display basic statistics of dataset: `shape` and `disbalance` for each subset.

        Parameters
        ----------
        `bounds` : list
            the boundaries along which the data will be sliced

        `show_split` : bool
            the parameter responsible for whether to show the data after slicing
        """

        if self.time_col is not None:
            self.dataset = self.dataset.sort_values(self.time_col, ascending=True).reset_index(drop=True).copy()
            strategic = 'index'
        
        self.dataset[self._split_col] = Splitter(bounds, self._bounds_mapping, strategic).split(self.dataset) 

        # show statistic per subset
        self._get_split_statistics()

        if show_split:
            self._show_splits()


    def _show_splits(self, n_ticks=10):
        """
        Plot the graph (if `time_col` is not empty) of splitted dataset
        """
        if self.time_col is not None:
            plt.figure(figsize=(12, 4))

            if self.time_col_type == 'int':
                df_to_plot = self.dataset[[self.time_col, self._split_col]] \
                                 .rename(columns={self._split_col: 'Subsets of data'})

                sns.histplot(df_to_plot, x=self.time_col,
                             bins=df_to_plot.shape[0] // 1000,
                             hue='Subsets of data', multiple="stack")
               
            elif self.time_col_type == 'datetime':
                data_for_plot = self.dataset[[self.time_col, self._split_col, self.target_col]] \
                                    .rename(columns={self._split_col: 'Subsets of data'})
                data_for_plot[self.time_col] = pd.to_datetime(data_for_plot[self.time_col]).dt.strftime('%Y-%m-%d %Hh')

                data_for_plot = data_for_plot.groupby([self.time_col, 'Subsets of data']) \
                                             .agg(cnt_events=(self.target_col, 'count')).reset_index() \
                                             .pivot(index=self.time_col, columns='Subsets of data', values='cnt_events')
                
                step = data_for_plot.shape[0] // n_ticks
                ticks = range(0, len(data_for_plot.index), step)
                labels = data_for_plot.index[::step]

                data_for_plot.plot.bar(stacked=True, figsize=(14, 4))
                plt.xticks(ticks, labels, rotation=45)

            plt.title('Splitted data into 4 groups')
            plt.xlabel('Time axis')
            plt.ylabel('Number of samples')

    
    def _get_split_statistics(self):
        """
        Showing basic statistics of dataset: `shape` and `disbalance` for each subset
        """
        for group in self._bounds_mapping:
            print(self._get_statistic_per_group(group))


    def _get_statistic_per_group(self, group):
        """
        Showing basic statistics of dataset: `shape` and `disbalance`
        """
        group_shape = self.dataset[self.dataset[self._split_col] == self._bounds_mapping[group]].shape
        fraud_cnt = self.dataset[
            (self.dataset[self._split_col] == self._bounds_mapping[group])
            & (self.dataset[self.target_col] == 1)
            ].shape[0]

        return f"{self._bounds_mapping[group]}: shape is {group_shape};" +\
               f" {(group_shape[0] / self.dataset.shape[0] * 100):.1f}% of dataset" +\
               f"\nDisbalance: {(fraud_cnt / group_shape[0] * 100):.1f}% \n"
    

    def _prepare_index(self, subset):
        indexes = (
            self.dataset[self.dataset[self._split_col] == subset]
            .index.to_numpy()
        )
        columns = self.dataset.drop(columns=self._service_columns, errors='ignore').columns.tolist()
        return indexes, columns



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

        inds, cols = self._prepare_index('D_init')

        X = self.dataset.loc[inds, cols]
        y = self.dataset.loc[inds, self.target_col]

        self.M_relabler = DummyModel(preprocessor=self.preprocessor.pipe, **params).fit(X, y)

        # save current model
        self._save_model(self.M_relabler, self._model_relabler_path, 'model_relabler')

        # get statistics on validation subset
        roc_auc, ap = self._model_scores(self.M_relabler, 'D_control_1')
        print(f'\nROC_AUC: {roc_auc:.5f}')
        print(f'AP: {ap:.5f}\n')
        print(f"Best params: {self.M_relabler.best_params_}")



    def relabler_distribution(self, bins_step: float=0.01,
                                    subsets: list=['D_train'],
                                    figsize: tuple=(15, 5)):
        """
        Plot distribution of `subsets` data by target

        Parameters
        ----------
        `bin_step`
        `subset`
        """      
        _, axes = plt.subplots(1, len(subsets), sharex=True, figsize=figsize)
        plt.suptitle(f"Target distribution after using relabler on subsets")

        for i, sub in enumerate(subsets):
            inds, cols = self._prepare_index(sub)
        
            X = self.dataset.loc[inds, cols]
            y = self.dataset.loc[inds, self.target_col]

            y_proba = self.M_relabler.predict_proba(X)[:, 1]

            data_for_plot = pd.concat([y.reset_index(drop=True), pd.Series(y_proba)], axis=1) \
                              .set_axis([self.target_col, self._relable_score_col], axis=1)
            
            sns.histplot(data=data_for_plot, x=self._relable_score_col,
                         hue=self.target_col, binrange=(0, 1), binwidth=bins_step,
                         stat='percent', common_norm=False, ax=axes[i])
            
            y_labels = [f'{float(label.get_text())}%' for label in axes[i].yaxis.get_ticklabels()]
            axes[i].set_yticklabels(y_labels)
            axes[i].set_title(f"Target distribution score on '{sub}'")
            axes[i].set_xlabel('Score')
            axes[i].set_ylabel('Percent of data, %')


    def train_baseline(self, params: dict):
        """
        Train baseline model on 'D_train' part of dataset
        'params' is a dict-like object with required fields:

            - `model` - is a source model for training
            - `parameters` - grid of parameters for searching best algorithm
            - `scoring` - one or dict of scores
            - `refit`
            - `need_smote`
        """

        inds, cols = self._prepare_index('D_train')

        X = self.dataset.loc[inds, cols]
        y = self.dataset.loc[inds, self.target_col]

        self.M_baseline = DummyModel(preprocessor=self.preprocessor.pipe, **params).fit(X, y)

        # save current model
        self._save_model(self.M_baseline, self._model_baseline_path, 'model_baseline')

        # get statistics on validation subset
        roc_auc, ap = self._model_scores(self.M_baseline, 'D_control_1')
        print(f'\nROC_AUC: {roc_auc:.5f}')
        print(f'AP: {ap:.5f}\n')
        print(f"Best params: {self.M_baseline.best_params_}")
    

    def train_matrix(self, params: dict,
                              TH_legetim: dict,
                              TH_fraud: dict):
        """
        `TH_legetim` dict like {start: float, stop: float, step: float}
        `TH_fraud` dict like {start: float, stop: float, step: float}
        """
        inds, cols = self._prepare_index('D_train')

        self.TH_matrix = []
        model_number = 1

        n_steps = len(np.arange(**TH_legetim)) * len(np.arange(**TH_fraud))
        print(f"Number od steps: {n_steps}")

        for TH_l in np.arange(**TH_legetim):
            for TH_f in np.arange(**TH_fraud):
                print(f'Step {model_number} ... ')

                data = self._change_bounds(self.dataset.loc[inds, cols + [self.target_col]], TH_l, TH_f)
                X = data.loc[inds, cols]
                y = data.loc[inds, self._relable_target_col]

                disbalance = y.sum() / y.count()
                model = DummyModel(preprocessor=self.preprocessor.pipe, **params).fit(X, y)

                # get statistics on validation subset
                if (TH_l == 0) and (TH_f == 1):
                    model = self.M_baseline
                
                # save current model
                self._save_model(model, self._model_relabled_path, f'model_relabled_{model_number}')

                roc_auc, ap = self._model_scores(model, 'D_control_1')
                self.TH_matrix.append([f'model_relabled_{model_number}', TH_l, TH_f, roc_auc, ap, disbalance])

                model_number += 1
        
        # convert matrix to DataFrame
        self.TH_matrix = pd.DataFrame(self.TH_matrix, columns=['model_name', 'TH_l', 'TH_f', 'ROC_AUC', 'AP', 'Disbalance'])

        

    def _change_bounds(self, data, TH_l: float, TH_f: float):
        """
        """
        data[self._relable_score_col] = self.M_relabler.predict_proba(data)[:, 1]

        data[self._relable_target_col] = data[self.target_col]
        data.loc[data[self._relable_score_col] <= TH_l, self._relable_target_col] = 0
        data.loc[data[self._relable_score_col] >= TH_f, self._relable_target_col] = 1

        return data
    

    def _model_scores(self, model, subset):
        inds, cols = self._prepare_index(subset)

        X = self.dataset.loc[inds, cols]
        y = self.dataset.loc[inds, self.target_col]

        y_proba = model.predict_proba(X)[:, 1]
        roc_auc = roc_auc_score(y, y_proba)
        ap = average_precision_score(y, y_proba)

        return roc_auc, ap
    
    def display_model_curves(self, model: str,
                                   test_subset: str,
                                   train_subset: str=None,
                                   figsize: tuple=(15, 5)):
        """
        `model`: 2 pissible values:
            - 'relabler'
            - 'baseline'
        """
        names = {key: value for key, value in zip(['Score on train', 'Score on test'],
                                                  [train_subset, test_subset]) if value is not None}

        if model == 'Relabler':
                model_ = self.M_relabler
        elif model == 'Baseline':
            model_ = self.M_baseline

        _, axes = plt.subplots(1, 2, figsize=figsize)
        plt.suptitle(f"Scores curves for '{model}'")

        for name, subset in names.items():

            inds, cols = self._prepare_index(subset)
            X = self.dataset.loc[inds, cols]
            y = self.dataset.loc[inds, self.target_col]
            y_proba = model_.predict_proba(X)[:, 1]

            for i, data in enumerate(zip([PrecisionRecallDisplay, RocCurveDisplay],
                                         ['best', 'lower right'],
                                         ["Precision-Recall curve", "ROC-AUC curve"])):
                
                metric_curve, position, title = data
                __ = metric_curve.from_predictions(y, y_proba,
                                                   name=f'{name} ({subset})',
                                                   plot_chance_level=True if subset == test_subset else False,
                                                   ax=axes[i])

                axes[i].legend(loc=f'{position}')
                axes[i].set_title(title)


    def plot_TH_matrix(self, cmaps=['BuGn', 'RdYlGn'], figsize: tuple=(15, 5),
                       fmt: list=['.3f', '.3f'], with_relative: bool=False):
        """
        Plot heatmap based on various thresholds TH_l and TH_f after training model on relabled data `D_train`
        """
        title = 'Main metrics with different thresholds'
        title_relative = 'Difference of metrics between model with relabling and Baseline'

        self._plot_TH_matrix([0, 0], cmaps[0], False, title, figsize, fmt)
        
        if with_relative:
            relative_values = self._model_scores(self.M_baseline, 'D_control_1')[::-1]
            self._plot_TH_matrix(relative_values, cmaps[1], True, title_relative, figsize, fmt)


    def _plot_TH_matrix(self, relative_values: list, cmap, is_relative, title, figsize, fmt):
        """
        """
        metrics = ['AP', 'ROC_AUC']

        _, axes = plt.subplots(1, len(metrics), figsize=figsize)
        plt.suptitle(title)

        for i, metric in enumerate(metrics):
            scores = self.TH_matrix.pivot(index='TH_l', columns='TH_f', values=metric) - relative_values[i]

            if is_relative:
                sns.heatmap(scores.sort_index(ascending=False),
                            annot=True, cmap=cmap, center=0, fmt=fmt[i], ax=axes[i])
            else:
                sns.heatmap(scores.sort_index(ascending=False),
                            annot=True, cmap=cmap, fmt=fmt[i], ax=axes[i])
            
            x_labels = [f'{float(label.get_text()):.2f}' for label in axes[i].xaxis.get_ticklabels()]
            y_labels = [f'{float(label.get_text()):.2f}' for label in axes[i].yaxis.get_ticklabels()]

            axes[i].set_xticklabels(x_labels)
            axes[i].set_yticklabels(y_labels)

            axes[i].set_title(f'TH matrix for {metric}')


    def compare_models(self, method, main_metric, thresh=None, show_curve: bool=False, figsize: tuple=(15, 5)):
        """
        Compare statistics of M_b and M_s models 'D_control_2'

        Parameters
        ----------
        `method`: two possible values:
            - auto
            - exact

        `main_metric`: two possible values:
            - 'AP'
            - 'ROC_AUC'
        """
        if method == 'auto':
            model_to_compare, TH_l, TH_f, disbalance = self._choose_best(main_metric)

        elif method == 'exact':
            TH_l, TH_f = thresh
            chosed_model = self.TH_matrix[(self.TH_matrix.TH_l.apply(lambda x: f"{x:.3f}") == f"{TH_l:.3f}") 
                                          & (self.TH_matrix.TH_f.apply(lambda x: f"{x:.3f}") == f"{TH_f:.3f}")]

            model_to_compare = load(self._model_relabled_path + f'/{chosed_model.model_name.values[0]}.joblib')
            disbalance = chosed_model.Disbalance.values[0]

        models = [self.M_baseline, model_to_compare]

        disbalance_list = [self.dataset[self.dataset[self._split_col] == 'D_train'][self.target_col].sum() \
                                / self.dataset[self.dataset[self._split_col] == 'D_train'][self.target_col].count(),
                           disbalance]

        text_for_legend = ['Baseline', f'Baseline with TH_l={TH_l:.3f} and TH_f={TH_f:.3f}']

        for model, descr, disb in zip(models, text_for_legend, disbalance_list):
            roc_auc, ap = self._model_scores(model, 'D_control_2')

            print(f"Model: {descr}")
            print(f"Disbalance on 'D_train': {(disb * 100):.3f}%\n")

            print("Metrics on 'D_control_2':")
            print(f'ROC_AUC: {roc_auc:.5f}')
            print(f'AP: {ap:.5f}\n')
        
        if show_curve:
            self._compare_models_curves(models, 'D_control_2',
                                        ['Baseline', 'Baseline with TH-relabling'], figsize)


    def _compare_models_curves(self, models, subset, text_for_legend, figsize: tuple=(15, 5)):
        """
        """
        _, axes = plt.subplots(1, 2, figsize=figsize)
        plt.suptitle('Comparison of the main metrics for the models')

        for model, text in zip(models, text_for_legend):

            inds, cols = self._prepare_index(subset)
            X = self.dataset.loc[inds, cols]
            y = self.dataset.loc[inds, self.target_col]
            y_proba = model.predict_proba(X)[:, 1]

            for i, data in enumerate(zip([PrecisionRecallDisplay, RocCurveDisplay],
                                         ['best', 'lower right'],
                                         ["Precision-Recall curve", "ROC-AUC curve"])):
                
                metric_curve, position, title = data
                __ = metric_curve.from_predictions(y, y_proba,
                                                   name=f'{text}',
                                                   ax=axes[i])

                axes[i].legend(loc=f'{position}')
                axes[i].set_title(title)



    def _choose_best(self, metric) -> tuple:
        best_result = self.TH_matrix.sort_values(metric, ascending=False).head(1)

        TH_l, TH_f = best_result.TH_l.values[0], best_result.TH_f.values[0]
        best_model = load(self._model_relabled_path + f'/{best_result.model_name.values[0]}.joblib')
        disbalance = best_result.Disbalance.values[0]

        return best_model, TH_l, TH_f, disbalance
    

    def _save_model(self, object, path, name):
        """
        Save model `object` to the `path` with current `name`
        """
        if not os.path.exists(path):
            os.makedirs(path)
        dump(object, f"{path}/{name}.joblib")
    

    def __str__(self):
        cat_cols, numeric_cols = self.preprocessor.get_columns()

        fraud_cnt = self.dataset[(self.dataset[self.target_col] == 1)].shape[0]
        disbalance = fraud_cnt / self.dataset.shape[0] * 100

        return f"Data shape is {self.dataset.shape}, target column is '{self.target_col}'\n" +\
            f"Time column is '{self.time_col}'\n" +\
            f"Disbalance is {disbalance:.3f}%\n\n" +\
            "#" * 30 + '\n' +\
            f"Number of categorical columns is {len(cat_cols)}\n" +\
            f"Number of numeric columns is {len(numeric_cols)}\n" +\
            "#" * 30 + '\n\n' +\
            f"Categorical columns: {cat_cols}\n" +\
            f"Numeric columns: {numeric_cols}"
