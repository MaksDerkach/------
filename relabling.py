import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class DatasetRelable:
    def __init__(self, dataset: pd.DataFrame, target_col: str, cat_cols: list=None,
                 numeric_cols: list=None, ordinal_cols=None) -> None:
        self.dataset = dataset
        self.target_col = target_col

        self.cat_cols = cat_cols
        self.numeric_cols = numeric_cols
        self.ordinal_cols = ordinal_cols

    def describe(self):
        print(f"Data shape is {self.dataset.shape}, target column is '{self.target_col}'")
        print(f"Categorical columns: {self.cat_cols}")
        print(f"Numeric columns: {self.numeric_cols}")
        print(f"Ordinal columns: {self.ordinal_cols}")

    def auto_selection_dtypes(self):
        pass


    def plot_data(self, time_col):
        pass

    def data_info(self):
        pass

    def basic_fit(self, test_size=0.2):
        pass
    
    def pca(self):
        pass

    def set_pipeline(self):
        pass
    
