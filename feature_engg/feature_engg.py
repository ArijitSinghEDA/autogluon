import featuretools as ft
# data = ft.demo.load_mock_customer()
# customers_df = data["customers"]
# print(customers_df)
from typing import List, Union
from sklearn.datasets import load_iris
import pandas as pd
pd.set_option('display.max_columns', None)

class feature_tool_pipeline:

    def __init__(self, data, id:str, trans_primitives: List[str]):
        self.data = data
        self.id = id
        self.trans_primitives = trans_primitives
    
    def generate_features(self):

        # if type(self.data) == str:
        #         df = pd.read_csv(self.data)
        # else:
        #     iris = self.data
        #     df = pd.DataFrame(iris.data, columns=iris.feature_names)
        #     df['species'] = iris.target
        #     df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        df = self.data

        es = ft.EntitySet(id = self.id)
        es = es.add_dataframe(
            dataframe_name="data",
            dataframe=df,
            index="index",
        )
        feature_matrix, feature_defs = ft.dfs(entityset = es, target_dataframe_name = 'data',
                                            # trans_primitives = ['add_numeric', 'multiply_numeric']
                                            trans_primitives = self.trans_primitives
                                            )

        return feature_matrix
# print(feature_defs[0])
# print(df.head())