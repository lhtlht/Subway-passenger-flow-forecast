import pandas as pd
import numpy as np




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

