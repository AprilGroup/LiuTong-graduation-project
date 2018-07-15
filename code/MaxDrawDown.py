
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from math import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import time
import dateutil,pylab  
from pylab import * 


# In[2]:


hs300 = pd.read_csv('E:/GraduationProject/hs300.csv', parse_dates=['Date'])
count = 10


# In[3]:


def CloseCurve(df):
    daytime = list(df["Date"])
    closelist = list(df["Close"])
    pylab.plot_date(pylab.date2num(daytime), closelist, marker='.', mfc='darkblue',linestyle='-')
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel("date") #X轴标签
    plt.ylabel("close") #Y轴标签
    plt.title("hs300 close curve") #标题
    plt.show()


# In[4]:


CloseCurve(hs300)


# In[5]:


#计算最大回撤率
def cal_max_drawdown(df,nav,sdate):
    maxdrawdown=(df[nav].iloc[0]-df[nav].iloc[1])/df[nav].iloc[0]
    start_date=df[sdate].iloc[0]
    end_date=df[sdate].iloc[1]
    for m in range(len(df)):
        for n in range(m+1,len(df)):
            if maxdrawdown<((df[nav].iloc[m]-df[nav].iloc[n])/df[nav].iloc[m]):
                maxdrawdown=(df[nav].iloc[m]-df[nav].iloc[n])/df[nav].iloc[m]
                start_date=df[sdate].iloc[m]
                end_date=df[sdate].iloc[n]

    print('最大回撤率是%g' % maxdrawdown)
    print(start_date,end_date)
    return maxdrawdown, start_date, end_date


# In[6]:


drawdown,start,end = cal_max_drawdown(hs300,'Close','Date')


# In[6]:


position = pd.read_csv('E:/GraduationProject/position_frame.csv', parse_dates=['date'])
maxdrawdown = pd.read_csv('E:/GraduationProject/maxdrawdown.csv')


# In[7]:


start_date = maxdrawdown['start_date'].iloc[0]
end_date = maxdrawdown['end_date'].iloc[0]
position1 = position[position.date>start_date]
position_use = position1[position1.date<end_date]


# In[8]:


PositionByVcode = position_use.groupby(by=['vcode'], as_index=False)['position'].mean()


# In[9]:


topn = int(len(PositionByVcode)/20)


# In[10]:


df = PositionByVcode.sort_values(by='position')


# In[11]:


v_position_choose = df[0:topn].reset_index(drop=True)


# In[12]:


v_position_choose


# In[13]:


result_daily = pd.read_csv('E:/GraduationProject/result_daily.csv', parse_dates=['NavDate'])
del result_daily['ID']


# In[14]:


result_daily1 = result_daily[result_daily.NavDate>start_date]
result_daily_use = result_daily1[result_daily1.NavDate<end_date].reset_index(drop=True)
result_daily_use[0:10]


# In[15]:


vportcode = []
drawdownlist = []
startlist =[]
endlist = []


# In[16]:


for v in tqdm(list(set(result_daily_use['PortCode']))):
    vdata = result_daily_use[result_daily_use['PortCode'] == v].reset_index(drop=True)
    if len(vdata)<50:
        continue
    else:
        maxdrawdown,start,end = cal_max_drawdown(vdata,'Nav','NavDate')
        vportcode.append(v)
        drawdownlist.append(maxdrawdown)
        startlist.append(start)
        endlist.append(end)
    


# In[19]:


new_drawdown = pd.DataFrame()
new_drawdown['PortCode'] = vportcode
new_drawdown['MaxDrawDown'] = drawdownlist
new_drawdown['StartDate'] = startlist
new_drawdown['EndDate'] = endlist
new_drawdown.to_csv("E:/GraduationProject/nav_maxdrawdown.csv",index=False)
new_drawdown = new_drawdown[new_drawdown.MaxDrawDown>0]


# In[20]:


df = new_drawdown.sort_values(by='MaxDrawDown')
v_navmaxdrawdown_choose = df[0:topn].reset_index(drop=True)
v_navmaxdrawdown_choose


# In[23]:


NavByVcode = result_daily_use.groupby(by=['PortCode'], as_index=False)['Nav'].mean()
df = NavByVcode.sort_values(by='Nav',ascending=False)
v_nav_choose = df[0:topn].reset_index(drop=True)
v_nav_choose

