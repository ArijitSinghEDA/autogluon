import os
from autogluon_tabular_class import AutogluonClassifier
from callback_class import AGCallBack

obj = AutogluonClassifier("train_titanic.csv")
obj.create_train_test_data(test_size=0)
obj.set_target_column("Survived")

try:
    print("Getting into AutogluonModels directory")
    autogluon_model_dir = [path for path in os.listdir(os.getcwd()) if path == "AutogluonModels"][0]
except:
    print("Making new model")
    try:
        os.remove("risk_model_graph_information.txt")
    except:
        pass
    finally:
        cb = AGCallBack()
        obj.select_predictor(keep_logs=False, verbose=0, callback=cb.model_callback)
else:
    print("Using existing model")
    model_dir = [path for path in os.listdir(autogluon_model_dir)][0]
    obj.select_predictor(save_path=f"{autogluon_model_dir}/{model_dir:}/")
finally:
    print("Predicting results")
    # Predicted Results
    # obj.predict_results().to_csv("titanic_compare_good.csv", index=False)
    # # Predicted Probabilities
    # obj.predict_probability().to_csv("titanic_compare_probability_good.csv", index=False)
    # print(obj.display_feature_importance())
