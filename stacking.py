import pandas as pd
import numpy as np
df = pd.DataFrame()

pre_list_in = []
pre_list_out = []

m1 = pd.read_csv("submit/subway_flow_ts.csv", encoding="utf-8")
m2 = pd.read_csv("submit/subway_flow_lgb.csv", encoding="utf-8")

pre_list_in.append(m1['inNums'])
pre_list_in.append(m2['inNums'])
pre_list_out.append(m1['outNums'])
pre_list_out.append(m2['outNums'])

df['stationID'] = m1['stationID']
df['startTime'] = m1['startTime']
df['endTime'] = m1['endTime']
df['inNums'] = np.array(pre_list_in).mean(axis=0)
df['outNums'] = np.array(pre_list_out).mean(axis=0)

df.to_csv("submit/subway_flow_tslgb.csv", index=False)




