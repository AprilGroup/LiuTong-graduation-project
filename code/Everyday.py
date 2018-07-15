# coding: utf-8

import pandas as pd
import numpy as np
from math import *
from tqdm import tqdm

N = 250

df1 = pd.read_csv('E:/GraduationProject/Xueqiu/nav_sample.csv', parse_dates=['NavDate'])
csi = pd.read_csv('E:/GraduationProject/Xueqiu/csi300.csv', parse_dates=['Date'])
df2 = df1[df1.NavDate != '2015-01-01']
df3 = df2[df2.NavDate != '2015-01-02']

groupby = df3.groupby('PortCode')

PortCode = []
NavDate = []
Nav = []
RiseRatio = []
absstd = []
CsiRiseRatio = []

for group in tqdm(groupby):
    port_code = group[0]
    group_data = group[1]

    #annual_abs_std
    rise_ratio = [group_data['Nav'].iloc[i + 1] / group_data['Nav'].iloc[i]-1 for i in range(len(group_data) - 1)]
    rise_ratio.insert(0,0)
    group_data['RiseRatio'] = rise_ratio
    abs_std = np.nanstd(group_data['RiseRatio'])
    annual_abs_std = sqrt(N) * abs_std
    # print('绝对年化风险为：%f'%annual_abs_std)

    #annual_rel_std
    group_data = group_data.merge(csi, left_on='NavDate', right_on='Date', how='left')
    #csi_rise = group_data.Close.diff()
    #group_data['CsiRise'] = csi_rise
    csi_rise_ratio = [group_data['Close'].iloc[i + 1] / group_data['Close'].iloc[i]-1 for i in
                      range(len(group_data) - 1)]
    csi_rise_ratio.insert(0,0)
    group_data['CsiRiseRatio'] = csi_rise_ratio
    active_ratio = [rise_ratio[i] - csi_rise_ratio[i] for i in range(len(rise_ratio))]
    group_data['ActiveRatio'] = active_ratio
    rel_std = np.nanstd(group_data['ActiveRatio'])
    annual_rel_std = sqrt(N) * rel_std
    # print('相对年化风险为：%f'%annual_rel_std

    PortCode.extend(list(group_data['PortCode']))
    NavDate.extend(list(group_data['NavDate']))
    Nav.extend(list(group_data['Nav']))
    RiseRatio.extend(rise_ratio)
    CsiRiseRatio.extend(csi_rise_ratio)

result = pd.DataFrame()
result["PortCode"] = PortCode
result["NavDate"] = NavDate
result["Nav"] = Nav
result["RiseRatio"] = RiseRatio
result["CsiRiseRatio"] = CsiRiseRatio

result.to_csv("E:/GraduationProject/result_daily.csv",index=False)
