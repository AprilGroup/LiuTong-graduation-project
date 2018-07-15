
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from math import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import time

N = 250
start_date = '2016-06-30'


# In[2]:


df1 = pd.read_csv('E:/GraduationProject/Xueqiu/nav_sample.csv', parse_dates=['NavDate'])
csi = pd.read_csv('E:/GraduationProject/Xueqiu/csi300.csv', parse_dates=['Date'])
df2 = df1[df1.NavDate != '2015-01-01']
df3 = df2[df2.NavDate != '2015-01-02']


# In[3]:


def vindex(lastdate):#计算大V指标
    df = df3[df3.NavDate <= lastdate].reset_index(drop=True)
    groupby = df.groupby('PortCode')

    code = []
    relstd = []

    for group in groupby:
        port_code = group[0]
        group_data = group[1]
        start_date = group_data.iloc[0]['NavDate']
        end_date = group_data.iloc[-1]['NavDate']

        #annual_abs_nav
        start_nav = group_data.iloc[0]['Nav']
        end_nav = group_data.iloc[-1]['Nav']
        abs_nav = end_nav / start_nav
        dif = len(group_data)
        annual_abs_nav = pow(abs_nav, N / dif) - 1

        #annual_rel_nav
        start_csi = csi[csi['Date'] == start_date]['Close'].iloc[0]
        end_csi = csi[csi['Date'] == end_date]['Close'].iloc[0]
        ratio_csi = end_csi / start_csi
        annual_market_nav = pow(ratio_csi, N / dif) - 1
        annual_rel_nav = annual_abs_nav - annual_market_nav

        #annual_abs_std
        rise_ratio = [group_data['Nav'].iloc[i + 1] / group_data['Nav'].iloc[i]-1 for i in range(len(group_data) - 1)]
        rise_ratio.insert(0,float("nan"))
        group_data['RiseRatio'] = rise_ratio
        abs_std = np.nanstd(group_data['RiseRatio'])
        annual_abs_std = sqrt(N) * abs_std

        #annual_rel_std
        group_data = group_data.merge(csi, left_on='NavDate', right_on='Date', how='left')
        csi_rise_ratio = [group_data['Close'].iloc[i + 1] / group_data['Close'].iloc[i]-1 for i in
                          range(len(group_data) - 1)]
        csi_rise_ratio.append(float("nan"))
        group_data['CsiRiseRatio'] = csi_rise_ratio
        active_ratio = [rise_ratio[i] - csi_rise_ratio[i] for i in range(len(rise_ratio))]
        group_data['ActiveRatio'] = active_ratio
        rel_std = np.nanstd(group_data['ActiveRatio'])
        annual_rel_std = sqrt(N) * rel_std

        #sharpe ratio
        #rf = 0.03
        #rel_sr = annual_rel_nav / annual_rel_std

        code.append(port_code)
        relstd.append(annual_rel_std)

    return(code,relstd)


# In[4]:


timelist = csi[csi.Date > start_date]
lentime = len(timelist)


# In[5]:


rel_sr = pd.DataFrame()
portcode,relstd = vindex('2017-07-01')
rel_sr['PortCode'] = portcode
rel_sr['2017-07-01'] = relstd 

for i in tqdm(range(lentime)):
    cal_date = timelist.iloc[i]['Date'].strftime("%Y-%m-%d")
    print(cal_date)
    portcode,relstd = vindex(cal_date)
    tmp = pd.DataFrame()
    tmp['PortCode'] = portcode
    tmp[cal_date] = relstd
    rel_sr = rel_sr.merge(tmp,on='PortCode',how='outer')
    
rel_sr.to_csv('E:/GraduationProject/test_rel_std.csv',index=False)

