import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation


#E initialize the data ingestion configuration

@dataclass
class DataIngestionconfig:
    train_data_path : str = os.path.join('artifacts', 'train.csv')
    raw_data_path : str = os.path.join('artifacts', 'raw.csv')

## creating the class for data ingestion
class DataIngestion():
    def __init__ (self):
        self.ingestion_config = DataIngestionconfig()
    

    def initiate_data_ingestion(self):
        logging.info("Data ingestion methods Started..!!!")
        try:
            df = pd.read_csv(os.path.join('data','sentiment_anlysis.csv')) # reading the original data file as a Dataframe
            logging.info('Dataset read as pandas DataFrame')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True) # if the directory exist it does nothing, if its not there, it creates newone
            df.to_csv(self.ingestion_config.raw_data_path, index = False)

            logging.info('Train_Data_Path')
            
            df.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            
            logging.info('ingestion of data is completed')

            return(self.ingestion_config.train_data_path)
        
        except Exception as e:
            logging.info('Exception occured as Data ingestion stage')
            raise CustomException(e, sys)
