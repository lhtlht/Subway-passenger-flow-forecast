import datetime
import pandas as pd
import numpy as np
TRAIN_DATA_PATH = "./data/Metro_train/"
TEST_DATA_PATH = "./data/Metro_testA/"

def count_out(out_array):
    return out_array.count() - out_array.sum()

def inout_station(recoreds):
    recoreds.index = pd.DatetimeIndex(recoreds["time"])
    in_station = recoreds[['stationID', 'status']].groupby('stationID').resample('10min', closed="left", label="left").sum().rename(columns={'status': 'inNums'})[['inNums']].reset_index()
    in_station.rename(columns={'time': 'startTime'}, inplace=True)
    out_station = recoreds[['stationID', 'status']].groupby('stationID').resample('10min', closed="left", label="right").apply(count_out).rename(columns={'status': 'outNums'})[['outNums']].reset_index()
    out_station.rename(columns={'time': 'endTime'}, inplace=True)
    out_station = out_station.drop(columns=['stationID'])
    inout_station_data = pd.concat([in_station, out_station], axis=1)
    return inout_station_data

def data_train_processing():
    datestart = datetime.datetime.strptime("2019-01-01", "%Y-%m-%d")
    date_list = []
    for i in range(25):
        date_list.append(str(datestart)[0:10])
        datestart += datetime.timedelta(days=+1)
    date_list.append("2019-01-28")
    inout_train = pd.DataFrame()
    for record_date in date_list:
        if record_date == "2019-01-28":
            recoreds = pd.read_csv(TEST_DATA_PATH + f'testA_record_{record_date}.csv', encoding="utf-8")
        else:
            recoreds = pd.read_csv(TRAIN_DATA_PATH + f'record_{record_date}.csv', encoding="utf-8")
        inout_station_data = inout_station(recoreds)
        print(record_date, inout_station_data.shape)
        inout_train = inout_train.append([inout_station_data])
        print(inout_train.shape)
    print(inout_train.columns)
    inout_train[['stationID', 'startTime', 'endTime', 'inNums', 'outNums']].to_csv("temp_data/inout_train.csv", index=False)


if __name__ == "__main__":

    data_train_processing()
