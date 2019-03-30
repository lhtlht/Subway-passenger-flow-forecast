import pandas as pd
import numpy as np
import time
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold
class Stacking(object):
    def __init__(self, n_fold=10):
        self.n_fold = n_fold

    def get_stacking(self, oof_list, prediction_list, labels):
        train_stack = np.vstack(oof_list).transpose()
        test_stack = np.vstack(prediction_list).transpose()

        repeats = len(oof_list)
        #RepeatedKFold  p次k折交叉验证
        kfolder = RepeatedKFold(n_splits=self.n_fold, n_repeats=repeats, random_state=4590)
        kfold = kfolder.split(train_stack, labels)
        preds_list = list()
        stacking_oof = np.zeros(train_stack.shape[0])

        for train_index, vali_index in kfold:
            k_x_train = train_stack[train_index]
            k_y_train = labels.loc[train_index]
            k_x_vali = train_stack[vali_index]

            gbm = BayesianRidge(normalize=True)
            gbm.fit(k_x_train, k_y_train)

            k_pred = gbm.predict(k_x_vali)
            stacking_oof[vali_index] = k_pred

            preds = gbm.predict(test_stack)
            preds_list.append(preds)

        fold_mae_error = mean_absolute_error(labels, stacking_oof)
        print(f'stacking fold mae error is {fold_mae_error}')
        fold_score = 1 / (1 + fold_mae_error)
        print(f'fold score is {fold_score}')

        preds_columns = ['preds_{id}'.format(id=i) for i in range(self.n_fold * repeats)]
        preds_df = pd.DataFrame(data=preds_list)
        preds_df = preds_df.T
        preds_df.columns = preds_columns
        stacking_prediction = list(preds_df.mean(axis=1))

        return stacking_oof, stacking_prediction

def stacking_mean():
    df = pd.DataFrame()

    pre_list_in = []
    pre_list_out = []

    m1 = pd.read_csv("submit/subway_flow_ts.csv", encoding="utf-8")
    m2 = pd.read_csv("submit/subway_flow_lgb_v3.csv", encoding="utf-8")
    m3 = pd.read_csv("submit/subway_flow_xgb_v3.csv", encoding="utf-8")

    pre_list_in.append(m1['inNums'])
    pre_list_in.append(m2['inNums'])
    pre_list_in.append(m3['inNums'])
    pre_list_out.append(m1['outNums'])
    pre_list_out.append(m2['outNums'])
    pre_list_out.append(m3['outNums'])

    df['stationID'] = m1['stationID']
    df['startTime'] = m1['startTime']
    df['endTime'] = m1['endTime']
    df['inNums'] = np.array(pre_list_in).mean(axis=0)
    df['outNums'] = np.array(pre_list_out).mean(axis=0)

    df.to_csv("submit/subway_flow_ts_lgb_xgb_0330_v1.csv", index=False)

def model_eval(predict_label, real_label):
    sum = 0
    l = len(predict_label)
    for i in range(l):
        sum += abs(predict_label[i] - real_label[i])
    return sum/len(predict_label)

if __name__ == "__main__":
    #stacking_mean()

    stacker = Stacking()
    reg_models = [ 'xgb', 'lgb', 'ctb']
    oof_list_in = list()
    predict_list_in = list()
    oof_list_out = list()
    predict_list_out = list()
    labels_in = list()
    labels_out = list()
    is_model = True
    for rm in reg_models:
        train_data = pd.read_csv(f"submit/subway_flow_{rm}_final_train.csv", encoding="utf-8")
        test_data = pd.read_csv(f"submit/subway_flow_{rm}_final.csv", encoding="utf-8")
        # 结果修正
        train_data.loc[train_data.inNums < 1, rm+'_prein'] = 0
        train_data.loc[train_data.outNums < 2, rm+'_preout'] = 0
        #train_data[rm+'_prein'] = train_data.apply(lambda row: round(row[rm+'_prein'], 0), axis=1)
        #train_data[rm+'_preout'] = train_data.apply(lambda row: round(row[rm+'_preout'], 0), axis=1)
        if is_model:
            test_data = train_data[train_data['date']=='2019-01-28']
            train_data = train_data[train_data['date']!='2019-01-28']
        oof_list_in.append(train_data[rm+'_prein'])
        oof_list_out.append(train_data[rm+'_preout'])
        if is_model:
            predict_list_in.append(test_data[rm + '_prein'])
            predict_list_out.append(test_data[rm + '_preout'])
        else:
            predict_list_in.append(test_data['inNums'])
            predict_list_out.append(test_data['outNums'])

    (stacking_oof, stacking_prediction_in) = stacker.get_stacking(oof_list_in, predict_list_in, train_data['inNums'])
    (stacking_oof, stacking_prediction_out) = stacker.get_stacking(oof_list_out, predict_list_out, train_data['outNums'])
    if is_model:
        print(test_data.columns)
        print(stacking_prediction_in)
        test_data.reset_index(inplace=True)
        in_mae = model_eval(stacking_prediction_in, test_data['inNums'])
        out_mae = model_eval(stacking_prediction_out, test_data['outNums'])
        print("in_mae", in_mae, "out_mae", out_mae, "in_out_mae", (in_mae + out_mae) / 2.0)
    else:
        test_stacking = test_data[['stationID', 'startTime', 'endTime']]
        test_stacking['inNums'] = stacking_prediction_in
        test_stacking['outNums'] = stacking_prediction_out
        # 结果修正
        test_stacking.loc[test_stacking.inNums < 1, 'inNums'] = 0
        test_stacking.loc[test_stacking.outNums < 2, 'outNums'] = 0
        test_stacking.to_csv("submit/subway_flow_lgb_xgb_stacking_v1.csv", index=False)



