from thyroid.exception import ThyroidException
from thyroid.logger import logging
from thyroid.predictor import ModelResolver
import pandas as pd
import numpy as np
from thyroid.utils import load_object
import os,sys
from datetime import datetime
from typing import Optional
PREDICTION_DIR="prediction"

import numpy as np


def columnsPreprocessing(df:pd.DataFrame)->Optional[pd.DataFrame]:
        try:
            drop_column_names = ['TBG','TSH_measured','T3_measured','TT4_measured','T4U_measured','FTI_measured','TBG_measured']
            logging.info(f"Dropping as these columns are indicators of value in the next column")
            df.drop(list(drop_column_names),axis=1,inplace=True)

            df['sex'] = df['sex'].map({'F' : 0, 'M' : 1})
            for column in df.columns:
                if  len(df[column].unique())==2:
                    df[column] = df[column].map({'f' : 0, 't' : 1})
                if len(df[column].unique())==1:
                    df[column] = df[column].map({'f' : 0, 't' : 1})


            df = pd.get_dummies(df, columns=['referral_source'])
            #logging.info(f"unique values of tumor: {df['tumor'].unique()}")
            return df
        except Exception as e:
            raise ThyroidException(e, sys)

def start_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR,exist_ok=True)
        logging.info(f"Creating model resolver object")
        model_resolver = ModelResolver(model_registry="saved_models")
        logging.info(f"Reading file :{input_file_path}")
        df = pd.read_csv(input_file_path)
        df = columnsPreprocessing(df=df)
        df.replace({"?":np.NAN},inplace=True)
        logging.info(f"df in batch prediction is : {df}")
        #df = pd.get_dummies(df, columns=['referral_source'])
        #validation
        
        logging.info(f"Loading transformer to transform dataset")
        transformer = load_object(file_path=model_resolver.get_latest_transformer_path())
        
        input_feature_names =  list(transformer.feature_names_in_)
        logging.info(f"input_feature_names : {input_feature_names}")
        input_arr = transformer.transform(df[input_feature_names])

        logging.info(f"Loading model to make prediction:{input_arr}")
        model = load_object(file_path=model_resolver.get_latest_model_path())
        prediction = model.predict(input_arr)
        prediction = prediction.astype(int)

        
        target_encoder = load_object(file_path=model_resolver.get_latest_target_encoder_path())

        cat_prediction = target_encoder.inverse_transform(prediction)

        df["prediction"]=prediction
        df["cat_pred"]=cat_prediction


        prediction_file_name = os.path.basename(input_file_path).replace(".csv",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path = os.path.join(PREDICTION_DIR,prediction_file_name)
        df.to_csv(prediction_file_path,index=False,header=True)
        return prediction_file_path
    except Exception as e:
        raise ThyroidException(e, sys)