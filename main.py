from feature_processing import *
import model
import datetime
import sys
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

TRAIN_DATA_PATH = "./data/Metro_train/"
TEST_DATA_PATH = "./data/Metro_testA/"
TEMP_DATA_PATH = "./temp_data/"

STATION_NUM = 81

def load_data():
    train = pd.read_csv(TEMP_DATA_PATH+"inout_train_full_paytype.csv", encoding="utf-8")
    station_info = pd.read_csv(TEMP_DATA_PATH + "station_fill.csv", encoding="utf-8")
    date_info = pd.read_csv(TEMP_DATA_PATH + "date_fill.csv", encoding="utf-8")
    train['date'] = train.apply(lambda row: row['startTime'].split(' ')[0], axis=1)
    train['time'] = train.apply(lambda row: row['startTime'].split(' ')[1], axis=1)
    train = train.merge(station_info, how="left", on=['stationID'])
    train = train.merge(date_info, how="left", on=['date'])

    test = pd.read_csv(TEST_DATA_PATH + "testA_submit_2019-01-29.csv")
    test['time'] = test.apply(lambda row: row['startTime'].split(' ')[1], axis=1)
    test['date'] = ['2019-01-29' for i in range(test.shape[0])]
    test = test.merge(station_info, how="left", on=['stationID'])
    test = test.merge(date_info, how="left", on=['date'])
    return train, test[["stationID","startTime","endTime","time", "date", "lineID", "lineSort", "shift", "is_shift",
                        "weather1", "weather2", "temperature_max", "temperature_min", "Wind"]]

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
    train_full = model_train
    model_train = model_train[model_train['date'] >= '2019-01-14']

    #reg_model_list = ['ts', 'lgb', 'xgb']
    reg_model_list = ['lgb']
    for rmodel in reg_model_list:
        if rmodel == 'ts':
            test = model.weightTimeModel(model_train, test, is_model)
        else:
            print("开始特征处理")
            model_train, test = feature_processing(model_train, test, train_full)
            print("开始训练")
            test = model.reg_model(model_train, test, rmodel, is_model)
        if is_model:
            in_mae = model_eval(test['inNums'], test['realInNums'])
            out_mae = model_eval(test['outNums'], test['realOutNums'])
            print("in_mae",in_mae,"out_mae",out_mae,"in_out_mae",(in_mae+out_mae)/2.0)
        else:
            features = ['stationID', 'startTime', 'endTime', 'inNums', 'outNums']
            test = test[features]
            test.to_csv(f"submit/subway_flow_{rmodel}.csv", encoding="utf-8", index=False)
"""
baseline:  offline-in_mae in_mae 14.574074074074074 out_mae 16.256601508916322 in_out_mae 15.415337791495197,线上15.1778
07开始统计-in_mae 13.989540466392318 out_mae 14.90809327846365 in_out_mae 14.448816872427983,线上13.2032

in_mae 13.763631687242798 out_mae 14.348079561042525 in_out_mae 14.055855624142662 周末加权 线上13.2889


in_mae 14.377143347050755 out_mae 15.093278463648835 in_out_mae 14.735210905349795
in_mae 14.362740054869684 out_mae 15.15886488340192 in_out_mae 14.760802469135802  加上shift
in_mae 14.357681755829905 out_mae 15.095250342935529 in_out_mae 14.726466049382717 加上is_shift
in_mae 14.1440329218107 out_mae 14.861882716049383 in_out_mae 14.502957818930042  结果纠正<1
http://www.tianqihoubao.com/lishi/hangzhou/month/201901.html  天气数据


lgb:in_mae 14.21656378600823 out_mae 14.504372427983538 in_out_mae 14.360468106995885
xgb:in_mae 14.160579561042525 out_mae 14.334362139917696 in_out_mae 14.24747085048011

xgb:
"""



