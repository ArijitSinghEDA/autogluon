import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autogluon_tabular_class import ClassifierGraphs


class AGCallBack:
    def __init__(self):
        self.val_score = []
        self.model_name = []
        self.order = 0
        self.processing = None
        self.y_preds = dict()
        self.y_probas = dict()
        self.obj = None

    def model_callback(self, model, fi, X_val, y_val, problem_type):
        self.val_score.append(model.val_score)
        self.model_name.append(model.name)
        self.order += 1
        fi = fi["importance"].apply(lambda x: np.round(x / fi['importance'].max(), 6))
        fi = fi[fi >= 0.001]
        y_pred = pd.Series(model.predict(X_val))
        if problem_type != "regression":
            if problem_type == "binary":
                classes = sorted(y_val.unique().tolist())
                try:
                    y_proba = pd.DataFrame({classes[1]: model.predict_proba(X_val)})
                except:
                    y_proba = pd.DataFrame(model.predict_proba(X_val), columns=y_val.unique().tolist())
                else:
                    y_proba[classes[0]] = 1 - y_proba[classes[1]]
            else:
                y_proba = pd.DataFrame(model.predict_proba(X_val), columns=sorted(y_val.unique().tolist()))
            for col in y_pred.unique().tolist():
                # temp = pd.DataFrame({f"Actual_{col}": y_val.apply(lambda x: 1 if x == col else 0).values.tolist()})
                # temp_pred = pd.DataFrame({"Predicted": y_pred.apply(lambda x: 1 if x == col else 0).values.tolist()})
                # self.y_preds[col] = pd.concat([temp, temp_pred], axis=1)
                self.y_preds[col] = pd.DataFrame(
                    {
                        f"Actual_{col}": y_val.apply(lambda x: 1 if x == col else 0).values.tolist(),
                        "Predicted": y_pred.apply(lambda x: 1 if x == col else 0).values.tolist()
                    }
                )

            for col in y_proba.columns.tolist():
                # temp = pd.DataFrame({f"Actual_{col}": y_val.apply(lambda x: 1 if x == col else 0).values.tolist()})
                # temp_proba = pd.DataFrame({"Predicted_Probability": y_proba[col].values.tolist()})
                # self.y_probas[col] = pd.concat([temp, temp_proba], axis=1)
                self.y_probas[col] = pd.DataFrame(
                    {
                        f"Actual_{col}": y_val.apply(lambda x: 1 if x == col else 0).values.tolist(),
                        "Predicted_Probability": y_proba[col].values.tolist()
                    }
                )

            self.obj = ClassifierGraphs(predicted_results=self.y_preds, probability_results=self.y_probas)
            self.obj.calculate_confusion_matrix_metrics()
            self.obj.calculate_cumulative_prediction_metrics()

            # # ROC-AUC
            # FPR, TPR = self.obj.roc_data_points()
            #
            # # PRC-AUC
            # Recall, Precision = self.obj.prc_data_points()
            #
            # # Gain
            # baseline, Gain = self.obj.gain_data_points()
            #
            # # Lift
            # baseline, Lift = self.obj.lift_data_points()
        else:
            baseline = list(range(len(y_val)))
            Actual = y_val.values.tolist()
            Predicted = y_pred.values.tolist()
            Residual = (y_val.values - y_pred.values).tolist()

            # Actual vs Predicted
            plt.plot(baseline, Actual)
            plt.plot(baseline, Predicted)
            plt.title("Actual vs Predicted")
            plt.ylabel("Actual/Predicted Graph")
            plt.show()

            plt.scatter(Predicted, Residual, marker='.')
            plt.plot([min(Predicted), max(Predicted)], [0, 0], lw=2)
            plt.title("Residual Graph")
            plt.xlabel("Predicted")
            plt.ylabel("Residual")
            plt.show()

        with open("app_model_graph_information.txt", mode="a") as file:
            file.write("\n========================================")
            file.write(f"\nOrder             : {self.order}")
            file.write(f"\nModel             : {model.name}")
            file.write(f"\nValidation Score  : {np.round(model.val_score, 6)}")
            file.write("\nFeature Importance:")
            file.write(f"\n{fi}")
            if problem_type != "regression":
                file.write("\nPoints for ROC-AUC stored in FPR and TPR")
                file.write("\nPoints for PRC-AUC stored in Recall and Precision")
                file.write("\nPoints for Gain stored in Baseline and Gain")
                file.write("\nPoints for Lift stored in Baseline and Lift")
            else:
                file.write("\nPoints for Actual vs. Predicted stored in Baseline and Predicted")
                file.write("\nPoints for Residual stored in Predicted and Residual")
            file.write("\n========================================")

    def data_processing_callback(self):
        self.processing = True
