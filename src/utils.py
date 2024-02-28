import os
import sys
import pickle
import numpu as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.exception import CustomException
from src.logger import logging

def save_object(filt_path, obj):
    try:
        dir_path = os.path.dirname(filt_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(filt_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(s, sys)

def evaluate_model(x_train, y_train, x_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(x_train, y_train) # train model
            y_test_pred = model.predict(x_test) # predict testing data
            test_model_score = r2_score(y_test, y_test_pred) #  get r2 score for train and test data
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        logging.info("Exception occured during model training")
        raise CustomException(e, sys)


def load_object(filt_path):
    try:
        with open(filt_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception occured in load_object function utils')
        raise CustomException(e, sys)


