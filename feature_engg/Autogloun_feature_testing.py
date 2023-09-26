
# from AutoFeature_Custom import featuretools_class
from autogloun_with_featuretools import AutogluonCustomFeatureGenerator
import os

obj = AutogluonCustomFeatureGenerator(label='Survived')

# feature_generator = featuretools_class(id='titanic')

# X_transform = feature_generator._fit_transform(X=obj.data)

obj.feature_gen(data='train_titanic.csv',id='titanic')

obj.create_train_test_split(test_size=0.2)

try:
    autogluon_model_dir = [path for path in os.listdir(os.getcwd()) if path == 'AutogluonModels'][0]
    # print(autogluon_model_dir)
    # 
except:
    obj.AutoGluonTrainer(obj.train_data, verbosity=2, keep_logs = True)

else:
    model_dir = [path for path in os.listdir(autogluon_model_dir)][0]
    path = f'{autogluon_model_dir}/{model_dir:}/'
    obj.AutoGluonTrainer(obj.train_data, path=path, verbosity=2, keep_logs=True)

# print(obj.AutoGluonPredictor())
