import pandas as pd
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
import os
from typing import Union
import warnings

warnings.filterwarnings("ignore")


class AutogluonTabular:
    def __init__(self, data_path: str):
        self.data = TabularDataset(data_path)
        self.label = None
        self.train_data = None
        self.test_data = None
        self.model = None

    def set_target_column(self, label: str):
        self.label = label

    def create_train_test_data(self, test_size: Union[int, float]):
        # Errors to be thrown
        if isinstance(test_size, float) and (test_size < 0 or test_size > 1):
            raise ValueError("Keep the float value between 0 to 1")
        if isinstance(test_size, int) and (test_size < 0):
            raise ValueError("Keep the int value greater than 0")

        # Functionality
        if isinstance(test_size, float):
            train_sample = self.data.shape[0] - int(test_size * self.data.shape[0])
        elif isinstance(test_size, int):
            train_sample = self.data.shape[0] - test_size
        self.train_data = self.data.sample(n=train_sample, random_state=0)
        self.test_data = self.data.drop(index=self.train_data.index.tolist()).copy()

    def select_predictor(self, save_path: str = None, keep_logs: bool = False, verbose: int = 2, callback=None):
        # Errors to be thrown
        if self.label is None:
            raise ValueError("Please select a target column !")

        # Functionality
        if save_path is None:
            if keep_logs:
                self.model = TabularPredictor(label=self.label, log_to_file=keep_logs, verbosity=verbose).fit(
                    self.train_data, trainer_callback=callback)
            else:
                self.model = TabularPredictor(label=self.label, verbosity=verbose).fit(self.train_data, trainer_callback=callback)
        elif save_path is not None:
            self.model = TabularPredictor.load(save_path)

    def predict_results(self) -> pd.DataFrame:
        # Errors to be thrown
        if self.label is None:
            raise ValueError("Please select a target column !")

        # Functionality
        pred = self.model.predict(self.test_data.drop(columns=[self.label]))
        return pred

    def display_feature_importance(self):
        df = self.model.feature_importance(feature_stage='transformed', include_confidence_band=False)
        fi = df[['importance']]
        fi = fi['importance'].apply(lambda x: np.round(x / fi['importance'].max(), 6))
        fi = fi[fi >= 0.001]
        return fi


class AutogluonClassifier(AutogluonTabular):
    def predict_probability(self) -> pd.DataFrame:
        # Errors to be thrown
        if self.label is None:
            raise ValueError("Please select a target column !")

        # Functionality
        prob_pred = self.model.predict_proba(self.test_data.drop(columns=[self.label]))
        probs = dict()
        for col in prob_pred.columns.tolist():
            temp = pd.DataFrame(
                {f"Actual_{col}": self.test_data[self.label].apply(lambda x: 1 if x == col else 0).values.tolist()})
            temp_proba = pd.DataFrame({"Predicted_Probability": prob_pred[col].values.tolist()})
            probs[col] = pd.concat([temp, temp_proba], axis=1)
        return probs

    def predict_results(self) -> pd.DataFrame:
        # Errors to be thrown
        if self.label is None:
            raise ValueError("Please select a target column !")

        # Functionality
        pred = self.model.predict(self.test_data.drop(columns=[self.label]))
        preds = dict()
        for col in pred.unique().tolist():
            temp = pd.DataFrame(
                {f"Actual_{col}": self.test_data[self.label].apply(lambda x: 1 if x == col else 0).values.tolist()})
            temp_pred = pd.DataFrame({"Predicted": pred.apply(lambda x: 1 if x == col else 0).values.tolist()})
            preds[col] = pd.concat([temp, temp_pred], axis=1)
        return preds


class ClassifierGraphs:
    def __init__(self, predicted_results: dict[str, pd.DataFrame], probability_results: dict[str, pd.DataFrame]):
        self.pred_datas = predicted_results
        self.prob_datas = probability_results
        self.thr = np.linspace(0, 1, 101)
        self.TPR = dict()
        self.FPR = dict()
        self.Precision = dict()
        self.splits = []
        self.gain_lift_dfs = dict()
        self.baseline = []
        self.gain = []
        self.lift = []

    def threshold_range(self, n: int = 100):
        self.thr = np.linspace(0, 1, num=n + 1)

    def calculate_confusion_matrix_metrics(self):
        if self.thr is None:
            raise ValueError("Please assign a list of thresholds first !")
        cl_TP = dict()
        cl_FP = dict()
        cl_FN = dict()
        cl_TN = dict()
        for key in self.prob_datas.keys():
            TP = []
            FP = []
            FN = []
            TN = []
            prob_data = self.prob_datas[key]
            for th in self.thr:
                prob_data['Predicted'] = (prob_data['Predicted_Probability'] >= th).values.astype(int)
                TP.append(sum(((prob_data[f'Actual_{key}'] == 1) & (prob_data['Predicted'] == 1)).values.astype(int)))
                FN.append(sum(((prob_data[f'Actual_{key}'] == 1) & (prob_data['Predicted'] == 0)).values.astype(int)))
                FP.append(sum(((prob_data[f'Actual_{key}'] == 0) & (prob_data['Predicted'] == 1)).values.astype(int)))
                TN.append(sum(((prob_data[f'Actual_{key}'] == 0) & (prob_data['Predicted'] == 0)).values.astype(int)))
                # try:
                #     TPR.append(np.round(TP / (TP + FN), 6))
                # except ZeroDivisionError:
                #     TPR.append(1)
                # try:
                #     FPR.append(np.round(FP / (FP + TN), 6))
                # except ZeroDivisionError:
                #     FPR.append(1)
                # try:
                #     Precision.append(np.round(TP / (TP + FP), 6))
                # except ZeroDivisionError:
                #     Precision.append(1)
                prob_data.drop(columns=['Predicted'], inplace=True)
            cl_TP[key] = TP
            cl_FP[key] = FP
            cl_FN[key] = FN
            cl_TN[key] = TN
        self.aggregate_confusion_matrix_metrics(cl_TP, cl_FP, cl_FN, cl_TN)

    def aggregate_confusion_matrix_metrics(self, TP, FP, FN, TN):
        agg_TP = []
        agg_FP = []
        agg_FN = []
        agg_TN = []
        for key in TP.keys():
            agg_TP.append(TP[key])
            agg_FP.append(FP[key])
            agg_FN.append(FN[key])
            agg_TN.append(TN[key])
        avg_TP = np.average(agg_TP, axis=0)
        avg_FP = np.average(agg_FP, axis=0)
        avg_FN = np.average(agg_FN, axis=0)
        avg_TN = np.average(agg_TN, axis=0)
        self.TPR = list(np.round(avg_TP / np.sum([avg_TP, avg_FN], axis=0), 6))
        self.FPR = list(np.round(avg_FP / np.sum([avg_FP, avg_TN], axis=0), 6))
        self.Precision = list(np.round(np.nan_to_num(avg_TP / np.sum([avg_TP, avg_FP], axis=0), nan=1), 6))

    def data_splits(self, data: pd.DataFrame, n: int):
        self.splits = np.array_split(data, n)

    def calculate_cumulative_prediction_metrics(self, data_split: int = 10):
        for key in self.pred_datas.keys():
            pred_data = self.pred_datas[key]
            self.data_splits(pred_data, data_split)
            number_1s = []
            for smdata in self.splits:
                number_1s.append(smdata['Predicted'].sum())
            number_1s = sorted(number_1s, reverse=True)
            gain_lift_df = pd.DataFrame({"predicted_1s_bin": number_1s})
            gain_lift_df['cumulative_1s'] = gain_lift_df["predicted_1s_bin"].cumsum()
            gain_lift_df['gain'] = np.round(gain_lift_df['cumulative_1s'] / gain_lift_df['cumulative_1s'].values[-1], 6)
            gain_lift_df['random_model_score'] = np.round(
                gain_lift_df['cumulative_1s'].values[-1] / gain_lift_df.shape[0], 6)
            gain_lift_df['cumulative_random_model_score'] = gain_lift_df['random_model_score'].cumsum()
            gain_lift_df['lift'] = np.round(
                gain_lift_df['cumulative_1s'] / gain_lift_df['cumulative_random_model_score'], 6)
            gain_lift_df.drop(
                columns=['random_model_score', 'predicted_1s_bin', 'cumulative_1s', 'cumulative_random_model_score'],
                inplace=True)
            self.gain_lift_dfs[key] = gain_lift_df
        self.aggregate_cumulative_prediction_metrics()

    def aggregate_cumulative_prediction_metrics(self):
        agg_gain = []
        agg_lift = []
        for key in self.gain_lift_dfs.keys():
            if len(self.baseline) == 0:
                self.baseline = list(self.gain_lift_dfs[key].index+1)
            agg_gain.append(self.gain_lift_dfs[key]['gain'].values.tolist())
            agg_lift.append(self.gain_lift_dfs[key]['lift'].values.tolist())
        self.gain = list(np.average(agg_gain, axis=0))
        self.lift = list(np.average(agg_lift, axis=0))

    def roc_data_points(self) -> (list, list):
        return self.FPR, self.TPR

    def prc_data_points(self) -> (list, list):
        return self.TPR, self.Precision

    def gain_data_points(self) -> (list, list):
        return self.baseline, self.gain

    def lift_data_points(self) -> (list, list):
        return self.baseline, self.lift
