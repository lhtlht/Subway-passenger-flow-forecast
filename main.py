from feature_processing import *
import model
import datetime
import sys
import pandas as pd
import numpy as np

TRAIN_DATA_PATH = "./data/Metro_train/"
TEST_DATA_PATH = "./data/Metro_testA/"
TEMP_DATA_PATH = "./temp_data/"

STATION_NUM = 81

def load_data():
    train = pd.read_csv(TEMP_DATA_PATH+"inout_train_full.csv", encoding="utf-8")
    train['date'] = train.apply(lambda row: row['startTime'].split(' ')[0], axis=1)
    train['time'] = train.apply(lambda row: row['startTime'].split(' ')[1], axis=1)

    test = pd.read_csv(TEST_DATA_PATH + "testA_submit_2019-01-29.csv")
    test['time'] = test.apply(lambda row: row['startTime'].split(' ')[1], axis=1)
    return train, test[["stationID","startTime","endTime","time"]]

def model_eval(predict_label, real_label):
    sum = 0
    l = len(predict_label)
    for i in range(l):
        sum += abs(predict_label[i] - real_label[i])
    return sum/len(predict_label)


if __name__ == "__main__":
    #统计每个地铁站的时间流量
    '''
    data_train_processing()
    '''
    #训练模型
    train,test = load_data()
    is_model = True
    if is_model:
        model_test = train[train['date']=='2019-01-28']
        model_test.rename(columns={'inNums': 'realInNums', 'outNums': 'realOutNums'}, inplace=True)
        model_train = train[train['date'] != '2019-01-28']
        test = model_test
    else:
        model_train = train
    model_train = model_train[model_train['date']>='2019-01-14']

    #test = model.mean_model(model_train, test)
    #test = model.weightTimeModel(model_train, test, is_model)
    print("开始特征处理")
    model_train, test = feature_processing(model_train, test)
    print("开始训练")
    test = model.reg_model(model_train, test, is_model)
    if is_model:
        in_mae = model_eval(test['inNums'], test['realInNums'])
        out_mae = model_eval(test['outNums'], test['realOutNums'])
        print("in_mae",in_mae,"out_mae",out_mae,"in_out_mae",(in_mae+out_mae)/2.0)
    else:
        test.to_csv("submit/subway_flow_TimeSeries_v2.csv", encoding="utf-8", index=False)
"""
baseline:  offline-in_mae in_mae 14.574074074074074 out_mae 16.256601508916322 in_out_mae 15.415337791495197,线上15.1778
in_mae 13.989540466392318 out_mae 14.90809327846365 in_out_mae 14.448816872427983,线上13.2032

in_mae 13.763631687242798 out_mae 14.348079561042525 in_out_mae 14.055855624142662

"""



