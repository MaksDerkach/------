from imblearn.pipeline import Pipeline as Pipeline_imb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import numpy as np


class Preprocessor:
    def __init__(self, data):

        self.cat_cols, self.numeric_cols = self._auto_select_dtypes(data)
        self.pipe = self._preprocess_data(self.cat_cols, self.numeric_cols)


    def _auto_select_dtypes(self, X) -> list:
        """
        Automatic selection of data types for numeric and categorical columns

        return cat_col, numeric_cols
        """
        cat_cols = []
        numeric_cols = []     

        for col_name, dtype in X.dtypes.items():
            if type(dtype) in [np.dtypes.Int64DType, np.dtypes.Float64DType]:
                numeric_cols.append(col_name)

            elif type(dtype) in [np.dtypes.ObjectDType]:
                cat_cols.append(col_name)
        
        return cat_cols, numeric_cols


    def _preprocess_data(self, cat_cols, numeric_cols):
        """
        Preprocess the data with numeric and categorical values
        """
        numeric_transform = Pipeline(
            steps=[('imputer', SimpleImputer(strategy='median')),
                   ('scaler', StandardScaler())]
        )

        categorical_transform = Pipeline(
            steps=[('encoder', OneHotEncoder(handle_unknown='infrequent_if_exist'))]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('numeric', numeric_transform, numeric_cols),
                ('categorical', categorical_transform, cat_cols)
            ]
        )

        return preprocessor




class DummyModel:
    def __init__(self, model,
                       parameters,
                       preprocessor,
                       scoring,
                       refit=True,
                       needs_smote=True,
                       n_jobs=None,
                       verbose=2,
                       is_preprocessed=False):

        self.main_model = model
        self.parameters = {'model__' + key: values for key, values in parameters.items()}
        self.preprocessor = preprocessor
        self.scoring = scoring

        self.refit = refit
        self.needs_smote = needs_smote
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.is_preprocessed = is_preprocessed
    

    def fit(self, X, y):

        steps = [('model', self.main_model)]

        # if SMOTE is needed
        if self.needs_smote:
            steps = [('smote', SMOTE(random_state=42))] + steps
        
        # if data is not preprocessed
        if not self.is_preprocessed:
            steps = [('preprocessor', self.preprocessor)] + steps
        
        pipeline = Pipeline_imb(steps=steps)
            
        
        model = GridSearchCV(estimator=pipeline,
                            param_grid=self.parameters,
                            refit=self.refit,
                            scoring=self.scoring,
                            cv=3,
                            verbose=self.verbose,
                            n_jobs=self.n_jobs
                           ).fit(X, y)
        
        return model