import pymongo
from thyroid.utils import get_collection_as_dataframe
from thyroid.logger import logging
from thyroid.exception import ThyroidException
from thyroid.entity import config_entity
from thyroid.entity.config_entity import DataIngestionConfig
from thyroid.components.data_ingestion import DataIngestion
import sys,os

if __name__ =="__main__":
     try:
          #get_collection_as_dataframe(database_name="thyroid",collection_name="detectthyroid")
          training_pipeline_config = config_entity.TrainingPipelineConfig()
          data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
          print(data_ingestion_config.to_dict())
          data_ingestion= DataIngestion(data_ingestion_config=data_ingestion_config)
          print(data_ingestion.initiate_data_ingestion())
     except Exception  as e:
          print(e)
     

