from feature_processing import *
import datetime
import pandas as pd
import numpy as np

TRAIN_DATA_PATH = "./data/Metro_train/"
TEST_DATA_PATH = "./data/Metro_testA/"

TEMP_DATA_PATH = "./temp_data/"


def load_data():
    train = pd.read_csv(TEMP_DATA_PATH+"inout_train.csv", encoding="utf-8")
    train['date'] = train.apply(lambda row: row['startTime'].split(' ')[0], axis=1)
    train['hour'] = train.apply(lambda row: row['startTime'][11:13], axis=1)
    print(train.head())
    return train


if __name__ == "__main__":

    train = load_data()
    model_train = train[train['date']!='2019-01-28']
    model_test = train[train['date']=='2019-01-28']

    print(model_test.head())
