import pandas as pd
import numpy as np
import sys
import math
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
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
    'num_leaves': 63,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0,
    'reg_lambda': 5,
    'verbose': -1,
    'random_state': 4590,
    'n_jobs': -1
}
xgb_params = {
        'booster': 'gbtree',
        'learning_rate': 0.01,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.8,
        'objective': 'reg:linear',
        'n_estimators': 10000,
        'min_child_weight': 3,
        'gamma': 0,
        'silent': True,
        'n_jobs': -1,
        'random_state': 4590,
        'reg_alpha': 2,
        'reg_lambda': 0.1,
        'alpha': 1,
        'verbose': 1
    }
ctb_params = {
    'n_estimators': 10000,
    'learning_rate': 0.01,
    'random_seed': 4590,
    'reg_lambda': 0.08,
    'subsample': 0.7,
    'bootstrap_type': 'Bernoulli',
    'boosting_type': 'Plain',
    'one_hot_max_size': 10,
    'rsm': 0.5,
    'leaf_estimation_iterations': 5,
    'use_best_model': True,
    'max_depth': 6,
    'verbose': -1,
    'thread_count': 4
}
rf_params = {
'n_estimators': 1000,
'n_jobs': 4,
'random_state': 2019
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
        test_week = 5
        test_date = 26
    else:
        test_week = 6
        test_date = 27
    train_date_list = []

    for date in train_dates:
        #train_date_list.append((test_date - int(date.split('-')[2])) * 2)

        if datetime.strptime(date,'%Y-%m-%d').weekday()==test_week:
            train_date_list.append((test_date - int(date.split('-')[2])) * 0.05)
        elif datetime.strptime(date,'%Y-%m-%d').weekday() in [5,6]:
            train_date_list.append((test_date - int(date.split('-')[2])) * 0.1)
        elif datetime.strptime(date,'%Y-%m-%d').weekday() in [0,1,2,3]:
            train_date_list.append((test_date - int(date.split('-')[2])) * 5)
        else:
            train_date_list.append((test_date - int(date.split('-')[2])) * 5)
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
    # 结果修正
    test.loc[test.inNums < 1, 'inNums'] = 0
    test.loc[test.outNums < 1, 'outNums'] = 0
    test['inNums'] = test.apply(lambda row: round(row['inNums'],0), axis=1)
    test['outNums'] = test.apply(lambda row: round(row['outNums'], 0), axis=1)
    test = test.drop(columns=['time'])
    test.fillna(0, inplace=True)
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



def reg_model(model_train, test, model_type, is_model):
    import lightgbm as lgb
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor
    model_train.reset_index(inplace=True)
    test.reset_index(inplace=True)
    if model_type == 'rf':
        model_train.fillna(0, inplace=True)
    """
    features = ['hour', 'minute', 'weekday',
                'shift', 'is_shift',
                'preInNums', 'preOutNums', 'inMax', 'outMax',
                 'inMax_14d',  'outMax_14d', '7d_14d_indiff', '7d_14d_outdiff',
                 'p0_inMax', 'p0_outMax', 'p2_inMax', 'p2_outMax',
                'p014_inMax', 'p014_outMax','p214_inMax', 'p214_outMax',
                'is_first1','is_first2','is_last1','is_last2']
    sts_feature = ['preInNums', 'preOutNums', 'inMax', 'outMax', 'inMin', 'outMin', 'inMean', 'outMean',]
    onehot_features = ['stationID', 'time', 'lineID',  'lineSort','lineSortD']
    """
    features_in = ['hour', 'minute', 'weekday','is_weekday', 'shift', 'is_shift',
                'preInNums', 'inMin','inMean','inMax',
                   'p0_inMax','p0_preInNums','p0_inMean','p0_inMin',
                   'p1_inMax', 'p1_preInNums', 'p1_inMean', 'p1_inMin',
                   'p2_inMax', 'p2_preInNums', 'p2_inMean', 'p2_inMin',
                   'p3_inMax', 'p3_preInNums', 'p3_inMean', 'p3_inMin',
                'is_first1','is_first2','is_last1','is_last2']
    features_out = ['hour', 'minute', 'weekday', 'is_weekday', 'shift', 'is_shift',
                   'preOutNums', 'outMin', 'outMean', 'outMax',
                    'preOutNums_14d', 'outMin_14d', 'outMean_14d', 'outMax_14d',
                    'p0_outMax', 'p0_preOutNums', 'p0_outMean', 'p0_outMin',
                    'p1_outMax', 'p1_preOutNums', 'p1_outMean', 'p1_outMin',
                    'p2_outMax', 'p2_preOutNums', 'p2_outMean', 'p2_outMin',
                    'p3_outMax', 'p3_preOutNums', 'p3_outMean', 'p3_outMin',
                   'is_first1', 'is_first2', 'is_last1', 'is_last2']
    sts_feature = ['preInNums', 'preOutNums', 'inMax', 'outMax', 'inMin', 'outMin', 'inMean', 'outMean',]
    onehot_features = ['stationID', 'time', 'lineID',  'lineSort','lineSortD']
    combine = pd.concat([model_train, test], axis=0)
    combine = multi_column_LabelEncoder(combine, onehot_features, rename=True)
    #one hot 处理
    onehoter = OneHotEncoder()
    X_onehot = onehoter.fit_transform(combine[onehot_features])
    train_x_onehot = X_onehot.tocsr()[:model_train.shape[0]].tocsr()
    test_x_onehot = X_onehot.tocsr()[model_train.shape[0]:].tocsr()

    train_x_original = combine[features_in][:model_train.shape[0]]
    test_x_original = combine[features_in][model_train.shape[0]:]
    print(train_x_original.shape)
    print(train_x_onehot.shape)
    train_x = sparse.hstack((train_x_onehot, train_x_original)).tocsr()
    test_x = sparse.hstack((test_x_onehot, test_x_original)).tocsr()

    #imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    #train_x = imp.fit_transform(train_x)

    # print(model_train[features].head())
    # train_x = model_train[features]
    train_y_in = model_train['inNums']
    train_y_out = model_train['outNums']
    # test_x = test[features]

    n_fold = 5
    count_fold = 0
    preds_list = list()
    oof_in = np.zeros(train_x.shape[0])
    kfolder = KFold(n_splits=n_fold, shuffle=True, random_state=2019)
    kfold = kfolder.split(train_x, train_y_in)
    for train_index, vali_index in kfold:
        print("training......fold",count_fold)
        count_fold = count_fold + 1
        k_x_train = train_x[train_index]
        k_y_train = train_y_in.loc[train_index]
        k_x_vali = train_x[vali_index]
        k_y_vali = train_y_in.loc[vali_index]
        if model_type == 'lgb':
            dtrain = lgb.Dataset(k_x_train, k_y_train)
            dvalid = lgb.Dataset(k_x_vali, k_y_vali, reference=dtrain)
            lgb_model = lgb.LGBMRegressor(**lgb_params)
            lgb_model = lgb_model.fit(k_x_train, k_y_train, eval_set=[(k_x_train, k_y_train), (k_x_vali, k_y_vali)],
                                      early_stopping_rounds=200, verbose=False, eval_metric="l2")
            k_pred = lgb_model.predict(k_x_vali, num_iteration=lgb_model.best_iteration_)
            pred = lgb_model.predict(test_x, num_iteration=lgb_model.best_iteration_)
        elif model_type == 'xgb':
            xgb_model = XGBRegressor(**xgb_params)
            xgb_model = xgb_model.fit(k_x_train, k_y_train, eval_set=[(k_x_train, k_y_train), (k_x_vali, k_y_vali)],
                                      early_stopping_rounds=200, verbose=False)
            k_pred = xgb_model.predict(k_x_vali)
            pred = xgb_model.predict(test_x)
        elif model_type == 'rf':
            rf_model = RandomForestRegressor(n_estimators=100, max_depth=3, criterion="mae",n_jobs=-1,random_state=2019)
            model = rf_model.fit(k_x_train, k_y_train)
            k_pred = rf_model.predict(k_x_vali)
            pred = rf_model.predict(test_x)
        preds_list.append(pred)
        oof_in[vali_index] = k_pred
    preds_columns = ['preds_{id}'.format(id=i) for i in range(n_fold)]
    preds_df = pd.DataFrame(data=preds_list)
    preds_df = preds_df.T
    preds_df.columns = preds_columns
    preds_in = list(preds_df.mean(axis=1))

#---------------------------------------------------------------------------------------------------------------------

    train_x_original = combine[features_out][:model_train.shape[0]]
    test_x_original = combine[features_out][model_train.shape[0]:]
    print(train_x_original.shape)
    print(train_x_onehot.shape)
    train_x = sparse.hstack((train_x_onehot, train_x_original)).tocsr()
    test_x = sparse.hstack((test_x_onehot, test_x_original)).tocsr()
    preds_list = list()
    oof_out = np.zeros(train_x.shape[0])
    kfolder = KFold(n_splits=n_fold, shuffle=True, random_state=2019)
    kfold = kfolder.split(train_x, train_y_out)
    for train_index, vali_index in kfold:
        print("training......fold",count_fold)
        count_fold = count_fold + 1
        k_x_train = train_x[train_index]
        k_y_train = train_y_out.loc[train_index]
        k_x_vali = train_x[vali_index]
        k_y_vali = train_y_out.loc[vali_index]
        dtrain = lgb.Dataset(k_x_train, k_y_train)
        dvalid = lgb.Dataset(k_x_vali, k_y_vali, reference=dtrain)

        if model_type == 'lgb':
            dtrain = lgb.Dataset(k_x_train, k_y_train)
            dvalid = lgb.Dataset(k_x_vali, k_y_vali, reference=dtrain)
            lgb_model = lgb.LGBMRegressor(**lgb_params)
            lgb_model = lgb_model.fit(k_x_train, k_y_train, eval_set=[(k_x_train, k_y_train), (k_x_vali, k_y_vali)],
                                      early_stopping_rounds=200, verbose=False, eval_metric="l2")
            k_pred = lgb_model.predict(k_x_vali, num_iteration=lgb_model.best_iteration_)
            pred = lgb_model.predict(test_x, num_iteration=lgb_model.best_iteration_)
        else:
            xgb_model = XGBRegressor(**xgb_params)
            xgb_model = xgb_model.fit(k_x_train, k_y_train, eval_set=[(k_x_train, k_y_train), (k_x_vali, k_y_vali)],
                                      early_stopping_rounds=200, verbose=False)
            k_pred = xgb_model.predict(k_x_vali)
            pred = xgb_model.predict(test_x)
        oof_out[vali_index] = k_pred
        preds_list.append(pred)

    preds_columns = ['preds_{id}'.format(id=i) for i in range(n_fold)]
    preds_df = pd.DataFrame(data=preds_list)
    preds_df = preds_df.T
    preds_df.columns = preds_columns
    preds_out = list(preds_df.mean(axis=1))
# ---------------------------------------------------------------------------------------------------------------------
    #输出结果
    test['inNums'] = preds_in
    test['outNums'] = preds_out
    #结果修正
    test.loc[test.inNums < 1,'inNums'] = 0
    test.loc[test.outNums < 2, 'outNums'] = 0
    #test['inNums'] = test.apply(lambda row: round(row['inNums'], 0), axis=1)
    #test['outNums'] = test.apply(lambda row: round(row['outNums'], 0), axis=1)

    if is_model:
        test = test[['stationID', 'startTime', 'endTime', 'inNums', 'outNums', 'realInNums', 'realOutNums']]
        train = 0
    else:
        train = pd.DataFrame()
        train['date'] = model_train['date']
        train['inNums'] = train_y_in
        train['outNums'] = train_y_out
        train[model_type+'_prein'] = oof_in
        train[model_type+'_preout'] = oof_out
        #train[model_type + '_preout'] = np.zeros(train.shape[0])
        test = test[['stationID', 'startTime', 'endTime', 'inNums', 'outNums']]
    test.fillna(0, inplace=True)
    return train, test


def reg_ctb_model(model_train, test, model_type, is_model):
    from catboost import CatBoostRegressor
    model_train.reset_index(inplace=True)
    test.reset_index(inplace=True)

    features = ['hour', 'minute', 'weekday',
                'shift', 'is_shift',
                ]
    sts_feature = ['preInNums', 'preOutNums', 'inMax', 'outMax', 'inMin', 'outMin', 'inMean', 'outMean',]
    onehot_features = ['stationID', 'time', 'lineID',  'lineSort','lineSortD']
    combine = pd.concat([model_train, test], axis=0)
    combine = multi_column_LabelEncoder(combine, onehot_features, rename=True)
    #one hot 处理
    onehoter = OneHotEncoder()
    X_onehot = onehoter.fit_transform(combine[onehot_features])
    train_x_onehot = X_onehot.tocsr()[:model_train.shape[0]].tocsr()
    test_x_onehot = X_onehot.tocsr()[model_train.shape[0]:].tocsr()

    train_x_original = combine[features][:model_train.shape[0]]
    test_x_original = combine[features][model_train.shape[0]:]
    print(train_x_original.shape)
    print(train_x_onehot.shape)
    train_x = sparse.hstack((train_x_onehot, train_x_original)).toarray()
    test_x = sparse.hstack((test_x_onehot, test_x_original)).toarray()
    print(type(train_x))

    # print(model_train[features].head())
    # train_x = model_train[features]
    train_y_in = model_train['inNums']
    train_y_out = model_train['outNums']
    # test_x = test[features]

    n_fold = 5
    count_fold = 0
    preds_list = list()
    oof_in = np.zeros(train_x.shape[0])
    kfolder = KFold(n_splits=n_fold, shuffle=True, random_state=2019)
    kfold = kfolder.split(train_x, train_y_in)
    for train_index, vali_index in kfold:
        print("training......fold",count_fold)
        count_fold = count_fold + 1
        k_x_train = train_x[train_index]
        k_y_train = train_y_in.loc[train_index]
        k_x_vali = train_x[vali_index]
        k_y_vali = train_y_in.loc[vali_index]

        ctb_model = CatBoostRegressor(**ctb_params)
        ctb_model = ctb_model.fit(k_x_train, k_y_train, eval_set=[(k_x_train, k_y_train), (k_x_vali, k_y_vali)],
                                  early_stopping_rounds=200, verbose=False)
        k_pred = ctb_model.predict(k_x_vali)
        pred = ctb_model.predict(test_x)

        preds_list.append(pred)
        oof_in[vali_index] = k_pred
    preds_columns = ['preds_{id}'.format(id=i) for i in range(n_fold)]
    preds_df = pd.DataFrame(data=preds_list)
    preds_df = preds_df.T
    preds_df.columns = preds_columns
    preds_in = list(preds_df.mean(axis=1))

#---------------------------------------------------------------------------------------------------------------------

    preds_list = list()
    oof_out = np.zeros(train_x.shape[0])
    kfolder = KFold(n_splits=n_fold, shuffle=True, random_state=2019)
    kfold = kfolder.split(train_x, train_y_out)
    for train_index, vali_index in kfold:
        print("training......fold",count_fold)
        count_fold = count_fold + 1
        k_x_train = train_x[train_index]
        k_y_train = train_y_out.loc[train_index]
        k_x_vali = train_x[vali_index]
        k_y_vali = train_y_out.loc[vali_index]

        ctb_model = CatBoostRegressor(**ctb_params)
        ctb_model = ctb_model.fit(k_x_train, k_y_train, eval_set=[(k_x_train, k_y_train), (k_x_vali, k_y_vali)],
                                  early_stopping_rounds=200, verbose=False)
        k_pred = ctb_model.predict(k_x_vali)
        pred = ctb_model.predict(test_x)

        oof_out[vali_index] = k_pred
        preds_list.append(pred)

    preds_columns = ['preds_{id}'.format(id=i) for i in range(n_fold)]
    preds_df = pd.DataFrame(data=preds_list)
    preds_df = preds_df.T
    preds_df.columns = preds_columns
    preds_out = list(preds_df.mean(axis=1))

    #输出结果
    test['inNums'] = preds_in
    test['outNums'] = preds_out
    #结果修正
    test.loc[test.inNums < 1,'inNums'] = 0
    test.loc[test.outNums < 2, 'outNums'] = 0
    #test['inNums'] = test.apply(lambda row: round(row['inNums'], 0), axis=1)
    #test['outNums'] = test.apply(lambda row: round(row['outNums'], 0), axis=1)

    if is_model:
        test = test[['stationID', 'startTime', 'endTime', 'inNums', 'outNums', 'realInNums', 'realOutNums']]
        train = 0
    else:
        train = pd.DataFrame()
        train['date'] = model_train['date']
        train['inNums'] = train_y_in
        train['outNums'] = train_y_out
        train[model_type+'_prein'] = oof_in
        train[model_type+'_preout'] = oof_out
        test = test[['stationID', 'startTime', 'endTime', 'inNums', 'outNums']]
    test.fillna(0, inplace=True)
    return train, test

