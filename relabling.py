import pandas as pd
import sklearn
import matplotlib.pyplot as plt


class DatasetRelable:
    def __init__(self, dataset, label) -> None:
        self.dataset = dataset
        self.label = label

        self.shape = self.dataset.drop(columns=self.label, axis=1).shape
        self.x = 0
        self.y = 1



    
