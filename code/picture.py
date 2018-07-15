
# coding: utf-8

# In[1]:


import seaborn as sns
import pandas as pd
import numpy as np
from math import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import time
import dateutil,pylab  
from pylab import * 
import tushare


# In[2]:


baseline = pd.read_csv("E:/GraduationProject/baseline1.csv", parse_dates=['NavDate'])
hs = pd.read_csv('E:/GraduationProject/Xueqiu/csi300.csv', parse_dates=['Date'])
df = pd.read_csv("E:/GraduationProject/adjweight_60.csv", parse_dates=['NavDate'])


# In[3]:


first = df['Nav'].iloc[0]
df['AdjNav'] = df['Nav']/first
df1 = df.merge(hs,left_on='NavDate',right_on='Date',how='left')
first1 = df1['Close'].iloc[0]
df1['AdjClose'] = df1['Close']/first1
df1[0:10]


# In[4]:


first3 = baseline['Nav'].iloc[0]
baseline['AdjNav'] = baseline['Nav']/first3
baseline1 = baseline.merge(hs,left_on='NavDate',right_on='Date',how='left')
first4 = baseline1['Close'].iloc[0]
baseline1['AdjClose'] = baseline1['Close']/first4
baseline1[0:10]


# In[11]:


daytime = list(df1["NavDate"])
navlist = list(df1["AdjNav"])
closelist = list(df1['AdjClose'])
baselinelist = list(baseline1["AdjNav"])
plt.figure(figsize=(10,6))
#pylab.plot_date(pylab.date2num(daytime), navlist,color='r',alpha=0.8,marker='',linestyle='-',label='optimization')
pylab.plot_date(pylab.date2num(daytime), baselinelist,color='r',alpha=0.8,marker='',linestyle='-',label='baseline')
pylab.plot_date(pylab.date2num(daytime), closelist,color='g',alpha=0.8,marker='',linestyle='-',label='hs300')
plt.subplots_adjust(bottom=0.15)
plt.xlabel("Date") #X轴标签
plt.ylabel("Nav") #Y轴标签
plt.grid(which= 'major')
plt.legend()
plt.title("Baseline Nav Curve") #标题
plt.show()


# In[6]:


N=250
start_date = df1.iloc[0]['NavDate']
end_date = df1.iloc[-1]['NavDate']

#annual_abs_nav
start_nav = df1.iloc[0]['AdjNav']
end_nav = df1.iloc[-1]['AdjNav']
abs_nav = end_nav / start_nav
dif = len(df1)
annual_abs_nav = pow(abs_nav, N / dif) - 1

#annual_rel_nav
start_csi = hs[hs['Date'] == start_date]['Close'].iloc[0]
end_csi = hs[hs['Date'] == end_date]['Close'].iloc[0]
ratio_csi = end_csi / start_csi
annual_market_nav = pow(ratio_csi, N / dif) - 1
annual_rel_nav = annual_abs_nav - annual_market_nav

#annual_abs_std
rise_ratio = [df1['Nav'].iloc[i + 1] / df1['Nav'].iloc[i]-1 for i in range(len(df1) - 1)]
rise_ratio.insert(0,float("nan"))
df1['RiseRatio'] = rise_ratio
abs_std = np.nanstd(df1['RiseRatio'])
annual_abs_std = sqrt(N) * abs_std

#annual_rel_std
group_data = df.merge(hs, left_on='NavDate', right_on='Date', how='left')
csi_rise_ratio = [group_data['Close'].iloc[i + 1] / group_data['Close'].iloc[i]-1 for i in
                  range(len(group_data) - 1)]
csi_rise_ratio.append(float("nan"))
group_data['CsiRiseRatio'] = csi_rise_ratio
active_ratio = [rise_ratio[i] - csi_rise_ratio[i] for i in range(len(rise_ratio))]
group_data['ActiveRatio'] = active_ratio
rel_std = np.nanstd(group_data['ActiveRatio'])
annual_rel_std = sqrt(N) * rel_std

#sharpe ratio
rf = 0.03
abs_sr = (annual_abs_nav - rf) / annual_abs_std
rel_sr = annual_rel_nav / annual_rel_std

print('绝对年化收益率为：%f'%annual_abs_nav)
print('相对年化收益率为：%f'%annual_rel_nav)
print('绝对年化风险为：%f'%annual_abs_std)
print('相对年化风险为：%f'%annual_rel_std)
print('绝对年化夏普率为：%f'%abs_sr)
print('相对年化夏普率为：%f'%rel_sr)


# In[18]:


#计算最大回撤和最大回撤回填天数
def GetMaxDrawdown(cum_NAV, col_name,date_name):
    cum_NAV['rel_max'] = pd.expanding_max(cum_NAV[col_name])
    cum_NAV['drawdown'] = 1 - cum_NAV[col_name] / cum_NAV['rel_max']
    # print cum_NAV
    maxdd = cum_NAV['drawdown'].max()
    idx_bottom = cum_NAV['drawdown'].argmax() # 最大回撤的低点的index
    idx_top = cum_NAV[col_name].loc[:idx_bottom].argmax() # 最大回撤的高点的index
    start_date = cum_NAV[date_name].loc[idx_top]
    end_date = cum_NAV[date_name].loc[idx_bottom]
    filling_dates = cum_NAV.loc[idx_bottom:].index[cum_NAV['drawdown'].loc[idx_bottom:] == 0]
    fillend_date_index = cum_NAV.index[-1]
    fill_date = cum_NAV[date_name].loc[fillend_date_index]
    if len(filling_dates) > 0:

        fill_date = filling_dates[0]

    filling_length = len(cum_NAV.loc[idx_bottom:fillend_date_index])

    print ('Max_top starts from: ', start_date, '\nMax_bottom ends on: ', end_date, '\nMax drawdown = ', maxdd)
    print ('Max filling up ends on: ', fill_date, '\nLasting ', filling_length, ' trading days')
    return start_date,end_date,maxdd,fill_date,filling_length

#计算大盘的最大回撤
start,end,maxdd,fill,length = GetMaxDrawdown(df1,'AdjNav','NavDate')

