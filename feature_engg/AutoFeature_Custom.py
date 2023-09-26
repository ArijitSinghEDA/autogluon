from pandas import DataFrame
from autogluon.features.generators import AbstractFeatureGenerator
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
import featuretools as ft
from sklearn.datasets import load_iris
import pandas as pd
from feature_engg import feature_tool_pipeline
from autogluon.common.features.types import R_INT, R_FLOAT, R_OBJECT,R_CATEGORY

class featuretools_class(AbstractFeatureGenerator):

    def __init__(self, id, **kwargs):
        super().__init__(**kwargs)
        self.id = id

    def _fit_transform(self, X, **kwargs) -> DataFrame:
        X_out = self._transform(X)
        print(X_out.shape)
        return X_out

    def _transform(self, X) -> DataFrame:
        ft = feature_tool_pipeline(data=X,id=self.id,trans_primitives=['add_numeric', 'multiply_numeric'])
        return ft.generate_features()
    
    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[R_INT, R_FLOAT, R_OBJECT,R_CATEGORY]) 


# featuretools_demo = featuretools_class(id='titanic', verbosity=3)
# X_transform = featuretools_demo._fit_transform(X='train_data.csv')
# print(X_transform.head(5))

