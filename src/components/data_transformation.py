import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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

    def get_data_transformation_object(self,text):
        try:            
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

    def initiate_data_transformation(self, train_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            print(train_df)

            logging.info('Read train data completed')
            logging.info(f'Train Dataframe head : \n {train_df.head().to_string()}')
            
            logging.info('Data Transformings initiated/Text cleaning')

            train_df['Description'] = train_df['Description'].apply(lambda x : self.get_data_transformation_object(x))
            

            drop_columns = ['Is_Response','User_ID','Browser_Used','Device_Used']

            x = train_df['Description']
            y = train_df['Is_Response'].apply(lambda x: 1 if x == 'happy' else 0)  # Assuming 'happy' is positive class

            #print("input_feature_train_df:", input_feature_train_df.head())
            #print("target_feature_train_df:", target_feature_train_df.head())


            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

            #print("x_train:", x_train)
            #print("y_train:", y_train)

            # Creating Bag of Words (BOW)
            
            cv = CountVectorizer(max_features=2500, ngram_range=(1,2))

            x_train = cv.fit_transform(x_train).toarray()
            x_test = cv.transform(x_test).toarray()

            ## Trnasformating using preprocessor obj
            logging.info("Applying preprocessing object on training datasets.")

            #print("Shape of x_train:", x_train.shape)
            #print("Shape of y_train:", x_test.shape)
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=cv)
            logging.info('Preprocessor pickle file saved')


            return (x_train, x_test, y_train, y_test)
        
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e, sys)






        