import sys
import os
import re
import string
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

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
            logging.info("Exception occurred in prediction")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, Description: str):
        self.Description = Description

    def get_data_clean_object(self, text):
        try:
            # Text cleaning
            text = text.lower()
            text = re.sub('\\[.*?/]', '', text)
            text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
            text = re.sub('\\w*\\d\\w*', '', text)
            return [text]
            logging.info('Cleaning Completed')

        except Exception as e:
            logging.info("Error in Data Transformation/Text Cleaning")
            raise CustomException(e, sys)

    def get_data_as_dataframe(self):
        try:
            Custom_data_input = self.Description
            
            clean_data = self.get_data_clean_object(Custom_data_input)

            # Assuming you want to create a DataFrame from 'clean_data'
            df = pd.DataFrame({'Description': [clean_data]})
            logging.info("Dataframe Gathered")

            return df

        except Exception as e:
            logging.info('Exception occurred in prediction pipeline')
            raise CustomException(e, sys)




if __name__ == '__main__':
    obj = CustomData("This is the best place to stay in downtown Denver. Everything about it is awesome.")
    data = obj.get_data_clean_object("This is the best place to stay in downtown Denver. Everything about it is awesome.")
    print(data)
 
    predict_pipeline=PredictPipeline()
    pred=predict_pipeline.predict(data)
    results=round(pred[0],2)
    print("-------------------------------------------------------------------------------------------------------------------------------------------------------")

    print(results)