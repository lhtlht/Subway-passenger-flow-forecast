import pandas as pd
import numpy as np
import sys

def get_weight_list(num_list, test_week):
    sum_l = sum(num_list)
    num_list2 = []
    for i in num_list:
        num_list2.append(sum_l/i)
    sum_l = sum(num_list2)
    return [i/sum_l for i in num_list2]

def add_date_weight(date, nums, date_w):
    return nums * date_w[date]

def mean_model(model_train, test, train_type="off_line"):
    time_in_mean = model_train[['time', 'stationID', 'inNums']].groupby(['stationID', 'time']).mean().reset_index()
    time_out_mean = model_train[['time', 'stationID', 'outNums']].groupby(['stationID', 'time']).mean().reset_index()

    test = test.merge(time_in_mean, how="left", on=["stationID", "time"])
    test = test.merge(time_out_mean, how="left", on=["stationID", "time"])
    test['inNums'] = test.apply(lambda row: round(round(row['inNums'],0),1), axis=1)
    test['outNums'] = test.apply(lambda row: round(round(row['outNums'], 0),1), axis=1)
    test = test.drop(columns=['time'])
    test.fillna(0, inplace=True)
    print(test.info())
    return test

def weightTimeModel(model_train, test, is_model):
    train_dates = list(model_train.groupby('date').size().index)
    if is_model:
        train_date_list = [28-int(date.split('-')[2]) for date in train_dates]
        test_week = 0
        test_date_str = '2019-01-28'
    else:
        train_date_list = [29-int(date.split('-')[2]) for date in train_dates]
        test_week = 1
        test_date_str = '2019-01-29'
    date_wl = get_weight_list(train_date_list, test_week)
    date_wd = {}
    for d,w in zip(train_dates,date_wl):
        date_wd[d] = w

    model_train['inNums'] = model_train.apply(lambda row: row['inNums']*date_wd[row['date']], axis=1)
    model_train['outNums'] = model_train.apply(lambda row: row['outNums'] * date_wd[row['date']], axis=1)
    time_in_mean = model_train[['time', 'stationID', 'inNums']].groupby(['stationID', 'time']).sum().reset_index()
    time_out_mean = model_train[['time', 'stationID', 'outNums']].groupby(['stationID', 'time']).sum().reset_index()

    test = test.merge(time_in_mean, how="left", on=["stationID", "time"])
    test = test.merge(time_out_mean, how="left", on=["stationID", "time"])

    test['inNums'] = test.apply(lambda row: round(round(row['inNums'],0),1), axis=1)
    test['outNums'] = test.apply(lambda row: round(round(row['outNums'], 0),1), axis=1)
    test = test.drop(columns=['time'])
    test.fillna(0, inplace=True)
    print(test.info())
    return test