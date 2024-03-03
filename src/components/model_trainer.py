import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__ (self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, x_train, x_test, y_train, y_test):
        try:
          
            model = RandomForestClassifier()

            model_report = evaluate_model(x_train, x_test, y_train, y_test, model)

            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            save_object(file_path = self.model_trainer_config.trained_model_file_path, obj = model)
        
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e, sys)
