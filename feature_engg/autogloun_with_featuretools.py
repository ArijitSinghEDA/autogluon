import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from typing import Union
from AutoFeature_Custom import featuretools_class


# fc = featuretools_class()


class AutogluonCustomFeatureGenerator:

    def __init__(self, label:str):
        self.data = None
        self.label = label
        self.train_data=None
        self.test_data= None
        self.model = None

    def feature_gen(self, id:str, data:str):
        self.data = TabularDataset(data)
        feature_generator = featuretools_class(id)
        self.data = feature_generator._fit_transform(X=self.data)


    def create_train_test_split(self, test_size:Union[float, int]):

        if self.data is None:
            raise ValueError("First Create the DataFrame")
        if isinstance(test_size, float) and (test_size<0 or test_size>1):
            raise ValueError("Enter test size between 0 and 1")
        if isinstance(test_size, int) and (test_size<0 or test_size>50):
            raise ValueError("Enter test size value between 0 and 50")
        if isinstance(test_size, str):
            raise ValueError("Enter Valid int or float value")
        
        #Functionality
        if isinstance(test_size, float):
            train_sample = self.data.shape[0] - int(self.data.shape[0]*test_size)
        if isinstance(test_size, int):
            train_sample = self.data.shape[0] - int(self.data.shape[0]*test_size/100)

        self.train_data = self.data.sample(n=train_sample)
        self.test_data = self.data.drop(self.train_data.index.tolist()).copy()

# Will create the featuretools pipeline from testing file to feature_generator param for data


    def AutoGluonTrainer(self, train_data, path=None, verbosity=2, keep_logs: bool= True):
        # train_data will come from featuretools class
        if path is None:
            if keep_logs:
                self.model= TabularPredictor(label=self.label, log_to_file=keep_logs, verbosity= verbosity).fit(train_data, feature_generator = None)
            else:
                self.model = TabularPredictor(label=self.label, verbosity= verbosity).fit(train_data,feature_generator = None)
        else:
            self.model= TabularPredictor.load(path)

    def AutoGluonPredictor(self):

        y_pred = self.model.predict(self.test_data.drop(columns=[self.label]))
        return y_pred


    

    