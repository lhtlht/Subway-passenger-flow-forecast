from feature_processing import *
import model
import datetime
import sys
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

TRAIN_DATA_PATH = "./data/Metro_train/"
TEST_DATA_PATH = "./data/Metro_testB/"
TEMP_DATA_PATH = "./temp_data/"

STATION_NUM = 81

def load_data():
    train = pd.read_csv(TEMP_DATA_PATH+"inout_train_full_b.csv", encoding="utf-8")
    station_info = pd.read_csv(TEMP_DATA_PATH + "station_fill.csv", encoding="utf-8")
    date_info = pd.read_csv(TEMP_DATA_PATH + "date_fill.csv", encoding="utf-8")
    train['date'] = train.apply(lambda row: row['startTime'].split(' ')[0], axis=1)
    train['time'] = train.apply(lambda row: row['startTime'].split(' ')[1], axis=1)
    train = train.merge(station_info, how="left", on=['stationID'])
    train = train.merge(date_info, how="left", on=['date'])

    test = pd.read_csv(TEST_DATA_PATH + "testB_submit_2019-01-27.csv")
    test['time'] = test.apply(lambda row: row['startTime'].split(' ')[1], axis=1)
    test['date'] = ['2019-01-27' for i in range(test.shape[0])]
    test = test.merge(station_info, how="left", on=['stationID'])
    test = test.merge(date_info, how="left", on=['date'])
    test = test.drop(columns=['inNums','outNums'])
    return train, test

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
    is_model_save = True
    if is_model:
        model_test = train[train['date']=='2019-01-26']
        model_test.rename(columns={'inNums': 'realInNums', 'outNums': 'realOutNums'}, inplace=True)
        model_train = train[train['date'] != '2019-01-26']
        test = model_test
    else:
        model_train = train
    train_full = model_train

    #reg_model_list = ['ts', 'lgb', 'xgb', 'ctb' ,'rf']
    reg_model_list = ['xgb']
    for rmodel in reg_model_list:
        if rmodel == 'ts':
            model_train = model_train[model_train['date'] >= '2019-01-12']
            test_save = model.weightTimeModel(model_train, test, is_model)
            #test_save = model.mean_model(model_train, test, is_model)
        else:
            model_train = model_train[model_train['date'] >= '2019-01-14']
            print("开始特征处理")
            if is_model_save:
                model_train, test = feature_processing(model_train, test, train_full)
                model_train.to_csv(TEMP_DATA_PATH+"TRAIN.csv", encoding="utf-8", index=False)
                test.to_csv(TEMP_DATA_PATH+"TEST.csv", encoding="utf-8", index=False)
            else:
                model_train = pd.read_csv(TEMP_DATA_PATH + "TRAIN.csv", encoding="utf-8")
                test = pd.read_csv(TEMP_DATA_PATH + "TEST.csv", encoding="utf-8")
            print("开始训练")
            if rmodel == 'ctb':
                train, test_save = model.reg_ctb_model(model_train, test, rmodel, is_model)
            else:
                train, test_save = model.reg_model(model_train, test, rmodel, is_model)
        if is_model:
            in_mae = model_eval(test_save['inNums'], test_save['realInNums'])
            out_mae = model_eval(test_save['outNums'], test_save['realOutNums'])
            print("in_mae",in_mae,"out_mae",out_mae,"in_out_mae",(in_mae+out_mae)/2.0)
        else:
            features = ['stationID', 'startTime', 'endTime', 'inNums', 'outNums']
            test_save = test_save[features]
            test_save.to_csv(f"submit_B/subway_flow_{rmodel}_b_v1.csv", encoding="utf-8", index=False)
            train.to_csv(f"submit_B/subway_flow_{rmodel}_final_b_v1.csv", encoding="utf-8", index=False)
"""
baseline:  offline-in_mae in_mae 14.574074074074074 out_mae 16.256601508916322 in_out_mae 15.415337791495197,线上15.1778
07开始统计-in_mae 13.989540466392318 out_mae 14.90809327846365 in_out_mae 14.448816872427983,线上13.2032

in_mae 13.763631687242798 out_mae 14.348079561042525 in_out_mae 14.055855624142662 周末加权 线上13.2889


in_mae 14.377143347050755 out_mae 15.093278463648835 in_out_mae 14.735210905349795
in_mae 14.362740054869684 out_mae 15.15886488340192 in_out_mae 14.760802469135802  加上shift
in_mae 14.357681755829905 out_mae 15.095250342935529 in_out_mae 14.726466049382717 加上is_shift
in_mae 14.1440329218107 out_mae 14.861882716049383 in_out_mae 14.502957818930042  结果纠正<1
http://www.tianqihoubao.com/lishi/hangzhou/month/201901.html  天气数据



lgb:in_mae 14.21656378600823 out_mae 14.504372427983538 in_out_mae 14.360468106995885 lgb+ts+xgb 12.84
xgb:in_mae 14.160579561042525 out_mae 14.334362139917696 in_out_mae 14.24747085048011 线上12.96 

lgb:('in_mae', 14.094221536351165, 'out_mae', 14.36531207133059, 'in_out_mae', 14.229766803840878)
in_mae 13.89849108367627 out_mae 14.258916323731139 in_out_mae 14.078703703703704
xgb:('in_mae', 14.120627572016462, 'out_mae', 14.219564471879286, 'in_out_mae', 14.170096021947874)
in_mae 13.952932098765432 out_mae 14.127914951989027 in_out_mae 14.040423525377228

ctb
in_mae 13.957661920631585 out_mae 14.131122052792142 in_out_mae 14.044391986711863

 
 
b榜
ts:in_mae 13.417009602194787 out_mae 16.226680384087793 in_out_mae 14.82184499314129 线上14.2464
in_mae 13.41863854595336 out_mae 16.224794238683128 in_out_mae 14.821716392318244
lgb: in_mae 14.60984213976438 out_mae 16.82232692075707 in_out_mae 15.716084530260726
in_mae 14.65047339824519 out_mae 16.77729511354243 in_out_mae 15.71388425589381

xgb:in_mae 15.325033898064921 out_mae 17.394781090353955 in_out_mae 16.359907494209438
"""



