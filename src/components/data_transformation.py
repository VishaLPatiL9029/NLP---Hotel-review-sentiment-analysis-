import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object
import re
import string

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self, text):
        try:
            logging.info('Data Transofrmation initiated/Text cleaning')

            # Text cleaning
            text = text.lower()
            text = re.sub('\\[.*?/]', '', text)
            text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
            text = re.sub('\\w*\\d\\w*','',text)
            return text
            logging.info('cleaning Completed')

        except Exception as e:
            logging.info("Error in Data Transformation/Text Cleaning")
            raise CustomException(e,sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe head : \n {train_df.head().to_string()}')
            logging.info(f'Train Dataframe head : \n {test_df.head().to_string()}')

            logging.info('Obtaining Preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Is_Response'
            drop_columns = [target_column_name,'User_ID','Browser_Used','Device_Used']

            input_feature_train_df = train_df.drop(columns = drop_columns, axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = drop_columns, axis = 1)
            target_feature_test_df = test_df[target_column_name]

            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(file_path = self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj)

            logging.info("preprocessor pickle file saved")

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
        
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e, sys)






        