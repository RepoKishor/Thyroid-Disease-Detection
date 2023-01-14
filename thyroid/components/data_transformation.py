from thyroid.entity import artifact_entity,config_entity
from thyroid.exception import ThyroidException
from thyroid.logger import logging
from typing import Optional
import os,sys 
from sklearn.pipeline import Pipeline
import pandas as pd
from thyroid import utils
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from thyroid.config import TARGET_COLUMN

class DataTransformation:


    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,
                    data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact
        except Exception as e:
            raise ThyroidException(e, sys)


    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            knn_imputer = KNNImputer(n_neighbors=3, weights='uniform',missing_values=np.nan)
            robust_scaler =  RobustScaler()
            pipeline = Pipeline(steps=[
                    ('Imputer',knn_imputer),
                    ('RobustScaler',robust_scaler)
                ])
            return pipeline
        except Exception as e:
            raise ThyroidException(e, sys)


    def columnsPreprocessing(self,df:pd.DataFrame)->Optional[pd.DataFrame]:
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

    def initiate_data_transformation(self,) -> artifact_entity.DataTransformationArtifact:
        try:
            #reading training and testing file
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            #selecting input feature for train and test dataframe
            input_feature_train_df=train_df.drop(TARGET_COLUMN,axis=1)
            input_feature_train_df = self.columnsPreprocessing(df=input_feature_train_df)
            input_feature_train_df.replace({"?":np.NAN},inplace=True)

            input_feature_test_df = test_df.drop(TARGET_COLUMN,axis=1)
            input_feature_test_df = self.columnsPreprocessing(df=input_feature_test_df)
            input_feature_test_df.replace({"?":np.NAN},inplace=True)

            #selecting target feature for train and test dataframe
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)

            #transformation on target columns
            target_feature_train_arr = label_encoder.transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)
            logging.info(f"input_feature_train_df<<<<<<<<<: {input_feature_train_df}")
            transformation_pipleine = DataTransformation.get_data_transformer_object()
            transformation_pipleine.fit(input_feature_train_df)

            #transforming input features
            input_feature_train_arr = transformation_pipleine.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipleine.transform(input_feature_test_df)
            

            #smt = SMOTETomek(sampling_strategy="auto")
            rndovrsample = RandomOverSampler(random_state=49)
            logging.info(f"Before resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}")
            input_feature_train_arr, target_feature_train_arr = rndovrsample.fit_resample(input_feature_train_arr, target_feature_train_arr)
            logging.info(f"After resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}")
            
            logging.info(f"Before resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}")
            input_feature_test_arr, target_feature_test_arr = rndovrsample.fit_resample(input_feature_test_arr, target_feature_test_arr)
            logging.info(f"After resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}")

            #target encoder
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr ]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]


            #save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path,
                                        array=test_arr)


            utils.save_object(file_path=self.data_transformation_config.transform_object_path,
             obj=transformation_pipleine)

            utils.save_object(file_path=self.data_transformation_config.target_encoder_path,
            obj=label_encoder)



            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path,
                target_encoder_path = self.data_transformation_config.target_encoder_path

            )

            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise ThyroidException(e, sys)