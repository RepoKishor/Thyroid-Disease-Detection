import pymongo
import pandas as pd
import json


# Provide the mongodb localhost url to connect python to mongodb.
client = pymongo.MongoClient("mongodb://localhost:27017/")
DATABASE_NAME = "thyroid"
COLLECTION_NAME="detectthyroid"
DATA_FILE_PATH = "/config/workspace/hypothyroid.csv"


if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and COlumns:{df.shape}")
    df.reset_index(drop=True, inplace=True)
    #convert the dataframe to json so that we can dump the data into mongoDB 
    json_record = list(json.loads(df.T.to_json()).values())

    print(json_record[0])
    #insert the converted json record to mongodb
    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)