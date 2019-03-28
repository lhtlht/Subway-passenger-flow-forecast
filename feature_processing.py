import datetime
import pandas as pd
import numpy as np
TRAIN_DATA_PATH = "./data/Metro_train/"
TEST_DATA_PATH = "./data/Metro_testA/"
STATION_NUM = 81
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
        data_full = data_train_processing_fill(record_date)
        inout_station_data = data_full.merge(inout_station_data, how="left", on=['stationID', 'startTime', 'endTime'])
        inout_station_data.fillna(0, inplace=True)
        print(record_date, inout_station_data.shape)
        inout_train = inout_train.append([inout_station_data])
        print(inout_train.shape)
    print(inout_train.columns)
    inout_train[['stationID', 'startTime', 'endTime', 'inNums', 'outNums']].to_csv("temp_data/inout_train_full.csv", index=False)


def data_train_processing_fill(date):
    date_add = (datetime.datetime.strptime(date,'%Y-%m-%d')+datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    df_starttime = pd.date_range(date, date_add, freq="10min", closed="left")
    df_endtime = pd.date_range(date, date_add, freq="10min", closed="right")
    df_station = np.ones((df_starttime.shape[0],))
    df = pd.DataFrame()
    for i in range(STATION_NUM):
        station_df = pd.DataFrame()
        station_df['stationID'] = df_station * i
        station_df['startTime'] = df_starttime
        station_df['endTime'] = df_endtime
        df = df.append(station_df)
    return df


def get_weight_list(num_list):
    sum_l = sum(num_list)
    num_list2 = []
    for i in num_list:
        num_list2.append(sum_l/i)
    sum_l = sum(num_list2)
    return [i/sum_l for i in num_list2]

def get_predate_sts(train_full, test, pre_day, gfeature):
    date_mean = pd.DataFrame()
    train_dates_list = list(train_full.groupby('date').size().index)
    print("date mean......", gfeature, pre_day)
    for train_date in train_dates_list[13:]:
        test_date = int(train_date.split('-')[2])
        test_date_pre = (datetime.datetime.strptime(train_date, '%Y-%m-%d')-datetime.timedelta(days=pre_day)).strftime("%Y-%m-%d")

        train_data_range = train_full[(train_full['date']>=test_date_pre) & (train_full['date']<train_date)]
        train_dates = list(train_data_range.groupby('date').size().index)
        train_date_list = []
        for date in train_dates:
            train_date_list.append((test_date - int(date.split('-')[2])) * 2)
        date_wl = get_weight_list(train_date_list)
        date_wd = {}
        for d, w in zip(train_dates, date_wl):
            date_wd[d] = w

        #计算最大值和最小值
        time_in_max = train_data_range[[gfeature, 'stationID', 'inNums']].groupby(['stationID', gfeature]).max().reset_index().rename(columns={'inNums': 'inMax'})
        time_in_min = train_data_range[[gfeature, 'stationID', 'inNums']].groupby(['stationID', gfeature]).min().reset_index().rename(columns={'inNums': 'inMin'})
        time_in_mean = train_data_range[[gfeature, 'stationID', 'inNums']].groupby(['stationID', gfeature]).mean().reset_index().rename(columns={'inNums': 'inMean'})
        time_out_max = train_data_range[[gfeature, 'stationID', 'outNums']].groupby(['stationID', gfeature]).max().reset_index().rename(columns={'outNums': 'outMax'})
        time_out_min = train_data_range[[gfeature, 'stationID', 'outNums']].groupby(['stationID', gfeature]).min().reset_index().rename(columns={'outNums': 'outMin'})
        time_out_mean = train_data_range[[gfeature, 'stationID', 'outNums']].groupby(['stationID', gfeature]).mean().reset_index().rename(columns={'outNums': 'outMean'})


        train_data_range['inNums'] = train_data_range.apply(lambda row: row['inNums'] * date_wd[row['date']], axis=1)
        train_data_range['outNums'] = train_data_range.apply(lambda row: row['outNums'] * date_wd[row['date']], axis=1)
        time_in_mean_w = train_data_range[[gfeature, 'stationID', 'inNums']].groupby(['stationID', gfeature]).sum().reset_index()
        time_in_mean_w.rename(columns={'inNums': 'preInNums'}, inplace=True)
        time_out_mean_w = train_data_range[[gfeature, 'stationID', 'outNums']].groupby(['stationID', gfeature]).sum().reset_index()
        time_out_mean_w.rename(columns={'outNums': 'preOutNums'}, inplace=True)

        df = train_full[train_full['date']==train_date].merge(time_in_mean_w, how="left", on=["stationID", gfeature])
        df = df.merge(time_out_mean_w, how="left", on=["stationID", gfeature])

        df = df.merge(time_out_max, how="left", on=["stationID", gfeature])
        df = df.merge(time_out_min, how="left", on=["stationID", gfeature])
        df = df.merge(time_out_mean, how="left", on=["stationID", gfeature])
        df = df.merge(time_in_max, how="left", on=["stationID", gfeature])
        df = df.merge(time_in_min, how="left", on=["stationID", gfeature])
        df = df.merge(time_in_mean, how="left", on=["stationID", gfeature])

        date_mean = date_mean.append(df)



    #统计测试集
    test_date = test['date'][0]
    test_date_pre = (datetime.datetime.strptime(test_date, '%Y-%m-%d') - datetime.timedelta(days=7)).strftime(
        "%Y-%m-%d")
    train_data_range = train_full[(train_full['date'] >= test_date_pre)]
    train_dates = list(train_data_range.groupby('date').size().index)
    train_date_list = []
    for date in train_dates:
        train_date_list.append((int(test_date.split('-')[2]) - int(date.split('-')[2])) * 2)
    date_wl = get_weight_list(train_date_list)
    date_wd = {}
    for d, w in zip(train_dates, date_wl):
        date_wd[d] = w
    time_in_max = train_data_range[[gfeature, 'stationID', 'inNums']].groupby(['stationID', gfeature]).max().reset_index().rename(columns={'inNums': 'inMax'})
    time_in_min = train_data_range[[gfeature, 'stationID', 'inNums']].groupby(['stationID', gfeature]).min().reset_index().rename(columns={'inNums': 'inMin'})
    time_in_mean = train_data_range[[gfeature, 'stationID', 'inNums']].groupby(['stationID', gfeature]).mean().reset_index().rename(columns={'inNums': 'inMean'})

    time_out_max = train_data_range[[gfeature, 'stationID', 'outNums']].groupby(['stationID', gfeature]).max().reset_index().rename(columns={'outNums': 'outMax'})
    time_out_min = train_data_range[[gfeature, 'stationID', 'outNums']].groupby(['stationID', gfeature]).min().reset_index().rename(columns={'outNums': 'outMin'})
    time_out_mean = train_data_range[[gfeature, 'stationID', 'outNums']].groupby(['stationID', gfeature]).mean().reset_index().rename(columns={'outNums': 'outMean'})


    train_data_range['inNums'] = train_data_range.apply(lambda row: row['inNums'] * date_wd[row['date']], axis=1)
    train_data_range['outNums'] = train_data_range.apply(lambda row: row['outNums'] * date_wd[row['date']], axis=1)
    time_in_mean_w = train_data_range[[gfeature, 'stationID', 'inNums']].groupby(['stationID', gfeature]).sum().reset_index()
    time_in_mean_w.rename(columns={'inNums': 'preInNums'}, inplace=True)
    time_out_mean_w = train_data_range[[gfeature, 'stationID', 'outNums']].groupby(
        ['stationID', gfeature]).sum().reset_index()
    time_out_mean_w.rename(columns={'outNums': 'preOutNums'}, inplace=True)

    test = test.merge(time_in_mean_w, how="left", on=["stationID", gfeature])
    test = test.merge(time_out_mean_w, how="left", on=["stationID", gfeature])
    test = test.merge(time_out_max, how="left", on=["stationID", gfeature])
    test = test.merge(time_out_min, how="left", on=["stationID", gfeature])
    test = test.merge(time_out_mean, how="left", on=["stationID", gfeature])
    test = test.merge(time_in_mean, how="left", on=["stationID", gfeature])
    test = test.merge(time_in_max, how="left", on=["stationID", gfeature])
    test = test.merge(time_in_min, how="left", on=["stationID", gfeature])
    return date_mean, test




def feature_processing(train, test, train_full):
    train.reset_index(inplace=True)
    test.reset_index(inplace=True)

    train_full['hour'] = train_full.apply(lambda row: int(row['time'].split(':')[0]), axis=1)

    train['hour'] = train.apply(lambda row: int(row['time'].split(':')[0]), axis=1)
    train['minute'] = train.apply(lambda row: int(row['time'].split(':')[1]), axis=1)
    train['date_int'] = train.apply(lambda row: int(row['date'].split('-')[2]), axis=1)
    train['weekday'] = train.apply(lambda row: datetime.datetime.strptime(row['date'],'%Y-%m-%d').weekday(), axis=1)
    train['is_weekday'] = train.apply(lambda row: 1 if datetime.datetime.strptime(row['date'], '%Y-%m-%d').weekday()==6 else 0, axis=1)

    test['hour'] = test.apply(lambda row: int(row['time'].split(':')[0]), axis=1)
    test['minute'] = test.apply(lambda row: int(row['time'].split(':')[2]), axis=1)
    test['date_int'] = test.apply(lambda row: int(row['date'].split('-')[2]), axis=1)
    test['weekday'] = test.apply(lambda row: datetime.datetime.strptime(row['date'], '%Y-%m-%d').weekday(), axis=1)
    test['is_weekday'] = test.apply(lambda row: 1 if datetime.datetime.strptime(row['date'], '%Y-%m-%d').weekday()==6 else 0, axis=1)

    """
        preInNums,preOutNums,inMax,outMax,inMin,outMin,inMean,outMean
    """
    train_hour, test_hour = get_predate_sts(train_full, test, pre_day=7, gfeature='hour')
    train, test_7d = get_predate_sts(train_full, test, pre_day=7, gfeature='time')
    sts_feature = ['preInNums', 'preOutNums', 'inMax', 'outMax', 'inMin', 'outMin', 'inMean', 'outMean']
    train_14d, test_14d = get_predate_sts(train_full, test, pre_day=14, gfeature='time')

    test = test_7d
    for sf in sts_feature:
        train[sf + '_14d'] = train_14d[sf]
        test[sf + '_14d'] = test_14d[sf]
    for sf in sts_feature:
        train[sf + '_hour'] = train_hour[sf]
        test[sf + '_hour'] = test_hour[sf]
    train['7d_14d_indiff'] = train['inMax']-train['inMax_14d']
    train['7d_14d_outdiff'] = train['outMax'] - train['outMax_14d']
    train['7d_14d_inrate'] = train['inMax'] / (train['inMax_14d'])
    train['7d_14d_outrate'] = train['outMax'] / (train['outMax_14d'])

    train['7d_14d_indiffm'] = train['inMean'] - train['inMean_14d']
    train['7d_14d_outdiffm'] = train['outMean'] - train['outMean_14d']
    train['7d_14d_inratem'] = train['inMean'] / (train['inMean_14d'])
    train['7d_14d_outratem'] = train['outMean'] / (train['outMean_14d'])


    test['7d_14d_indiff'] = test['inMax'] - test['inMax_14d']
    test['7d_14d_outdiff'] = test['outMax'] - test['outMax_14d']
    test['7d_14d_inrate'] = test['inMax'] / (test['inMax_14d'])
    test['7d_14d_outrate'] = test['outMax'] / (test['outMax_14d'])

    test['7d_14d_indiffm'] = test['inMean'] - test['inMean_14d']
    test['7d_14d_outdiffm'] = test['outMean'] - test['outMean_14d']
    test['7d_14d_inratem'] = test['inMean'] / (test['inMean_14d'])
    test['7d_14d_outratem'] = test['outMean'] / (test['outMean_14d'])


    return train, test


def processing_map():
    station_map = pd.read_csv("data/Metro_roadMap.csv", encoding="utf-8")
    print(station_map.columns)
    print(station_map.sum(axis=0))

if __name__ == "__main__":

    #data_train_processing()
    #processing_map()

    #计算每种类型的卡的进出人数
    pass