import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__ (self):
        pass

    def get_data_clean_object(text):
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


    def predict(self, features):
        try: 
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            
            model = load_object(model_path)

            data_clean = preprocessor.transform(features)

            pred = model.predict(data_clean)

            return pred

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, Description:str):
        self.Description = Description

    def get_data_as_dataframe(self):
        try:
            Custom_data_input_dict = {description : self.Description}
            clean_data = get_data_clean_object(Custom_data_input_dict)

            df = pd.DataFrame(clean_data)

            logging.info("Dataframe Gathered")

            return df


        except Exception as e:
            logging.info('Exception occured in prediction pipeline')
            raise CustomException(e,sys)
