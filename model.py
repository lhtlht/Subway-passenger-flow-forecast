import pandas as pd
import numpy as np
import sys
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'n_estimators': 10000,
    'metric': 'mae',
    'learning_rate': 0.01,
    'min_child_samples': 5,
    'min_child_weight': 0.01,
    'subsample_freq': 1,
    'num_leaves': 31,
    'max_depth': 5,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'reg_alpha': 0,
    'reg_lambda': 5,
    'verbose': -1,
    'seed': 4590
}

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
        test_week = 0
        test_date = 28
    else:
        test_week = 1
        test_date = 29
    train_date_list = []

    for date in train_dates:
        if datetime.strptime(date,'%Y-%m-%d').weekday()==test_week:
            train_date_list.append((test_date - int(date.split('-')[2])) * 1.5)
        elif datetime.strptime(date,'%Y-%m-%d').weekday()==6:
            train_date_list.append((test_date - int(date.split('-')[2])) * 3)
        elif datetime.strptime(date,'%Y-%m-%d').weekday()==5:
            train_date_list.append((test_date - int(date.split('-')[2])) * 2.5)
        else:
            train_date_list.append((test_date - int(date.split('-')[2])) * 2)
        #train_date_list.append((test_date - int(date.split('-')[2])) * (datetime.strptime(date,'%Y-%m-%d').weekday()-test_week+2))
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
    print(test.columns)
    test['inNums'] = test.apply(lambda row: round(row['inNums'],0), axis=1)
    test['outNums'] = test.apply(lambda row: round(row['outNums'], 0), axis=1)
    test = test.drop(columns=['time'])
    test.fillna(0, inplace=True)
    print(test.info())
    return test
def multi_column_LabelEncoder(df,columns,rename=True):
    le = LabelEncoder()
    for column in columns:
        print(column,"LabelEncoder......")
        le.fit(df[column])
        df[column+"_index"] = le.transform(df[column])
        if rename:
            df.drop([column], axis=1, inplace=True)
            df.rename(columns={column+"_index":column}, inplace=True)
    return df


def reg_model(model_train, test, is_model):
    import lightgbm as lgb
    model_train.reset_index(inplace=True)
    test.reset_index(inplace=True)
    print(model_train.head())

    features = [ 'date_int', 'hour', 'minute']
    onehot_features = ['stationID', 'date', 'time']
    combine = pd.concat([model_train, test], axis=0)
    combine = multi_column_LabelEncoder(combine, onehot_features, rename=True)
    #one hot 处理
    onehoter = OneHotEncoder()
    X_onehot = onehoter.fit_transform(combine[onehot_features])
    train_x_onehot = X_onehot.tocsr()[:model_train.shape[0]].tocsr()
    test_x_onehot = X_onehot.tocsr()[model_train.shape[0]:].tocsr()

    train_x_original = combine[features][:model_train.shape[0]]
    test_x_original = combine[features][model_train.shape[0]:]
    train_x = sparse.hstack((train_x_onehot, train_x_original)).tocsr()
    test_x = sparse.hstack((test_x_onehot, test_x_original)).tocsr()


    # print(model_train[features].head())
    # train_x = model_train[features]
    train_y_in = model_train['inNums']
    train_y_out = model_train['outNums']
    # test_x = test[features]

    n_fold = 5
    count_fold = 0
    preds_list = list()
    kfolder = KFold(n_splits=n_fold, shuffle=True, random_state=2019)
    kfold = kfolder.split(train_x, train_y_in)
    for train_index, vali_index in kfold:
        print(count_fold)
        count_fold = count_fold + 1
        k_x_train = train_x[train_index]
        k_y_train = train_y_in.loc[train_index]
        k_x_vali = train_x[vali_index]
        k_y_vali = train_y_in.loc[vali_index]
        dtrain = lgb.Dataset(k_x_train, k_y_train)
        dvalid = lgb.Dataset(k_x_vali, k_y_vali, reference=dtrain)

        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model = lgb_model.fit(k_x_train, k_y_train, eval_set=[(k_x_train, k_y_train), (k_x_vali, k_y_vali)],
                                  early_stopping_rounds=200, verbose=False, eval_metric="l2")
        k_pred_12 = lgb_model.predict(k_x_vali, num_iteration=lgb_model.best_iteration_)
        pred = lgb_model.predict(test_x, num_iteration=lgb_model.best_iteration_)

        preds_list.append(pred)

    preds_columns = ['preds_{id}'.format(id=i) for i in range(n_fold)]
    preds_df = pd.DataFrame(data=preds_list)
    preds_df = preds_df.T
    preds_df.columns = preds_columns
    preds_in = list(preds_df.mean(axis=1))

#---------------------------------------------------------------------------------------------------------------------

    preds_list = list()
    kfolder = KFold(n_splits=n_fold, shuffle=True, random_state=2019)
    kfold = kfolder.split(train_x, train_y_out)
    for train_index, vali_index in kfold:
        print(count_fold)
        count_fold = count_fold + 1
        k_x_train = train_x[train_index]
        k_y_train = train_y_in.loc[train_index]
        k_x_vali = train_x[vali_index]
        k_y_vali = train_y_in.loc[vali_index]
        dtrain = lgb.Dataset(k_x_train, k_y_train)
        dvalid = lgb.Dataset(k_x_vali, k_y_vali, reference=dtrain)

        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model = lgb_model.fit(k_x_train, k_y_train, eval_set=[(k_x_train, k_y_train), (k_x_vali, k_y_vali)],
                                  early_stopping_rounds=200, verbose=False, eval_metric="l2")
        k_pred_12 = lgb_model.predict(k_x_vali, num_iteration=lgb_model.best_iteration_)
        pred = lgb_model.predict(test_x, num_iteration=lgb_model.best_iteration_)

        preds_list.append(pred)

    preds_columns = ['preds_{id}'.format(id=i) for i in range(n_fold)]
    preds_df = pd.DataFrame(data=preds_list)
    preds_df = preds_df.T
    preds_df.columns = preds_columns
    preds_out = list(preds_df.mean(axis=1))

    #输出结果
    test['inNums'] = preds_in
    test['outNums'] = preds_out
    test['inNums'] = test.apply(lambda row: round(row['inNums'], 0), axis=1)
    test['outNums'] = test.apply(lambda row: round(row['outNums'], 0), axis=1)
    test = test.drop(columns=['time'])
    test.fillna(0, inplace=True)
    print(test.info())
    return test

