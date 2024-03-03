import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(s, sys)

def evaluate_model(x_train, x_test, y_train, y_test, model):
    try:
        model.fit(x_train, y_train)  # train model
        y_pred = model.predict(x_test)  # predict testing data
        test_model_score = accuracy_score(y_test,y_pred)  # get accuracy_score for train and test data

        return test_model_score

    except Exception as e:
        logging.info("Exception occurred during model training")
        raise CustomException(e, sys)


def load_object(filt_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception occured in load_object function utils')
        raise CustomException(e, sys)


