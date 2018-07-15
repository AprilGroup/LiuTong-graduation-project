
# coding: utf-8

# In[ ]:


import seaborn as sns
import pandas as pd
import numpy as np
from math import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import time

N = 250
train_date = '2015-12-31'


# In[ ]:


df1 = pd.read_csv('E:/GraduationProject/Xueqiu/nav_sample.csv', parse_dates=['NavDate'])
csi = pd.read_csv('E:/GraduationProject/Xueqiu/csi300.csv', parse_dates=['Date'])
df2 = df1[df1.NavDate != '2015-01-01']
df3 = df2[df2.NavDate != '2015-01-02']


# In[ ]:


def vindex(lastdate,toaddress):#计算大V指标
    df = df3[df3.NavDate <= lastdate].reset_index(drop=True)
    groupby = df.groupby('PortCode')

    code = []
    startday = []
    endday = []
    absnav = []
    relnav = []
    absstd = []
    relstd = []
    abssr = []
    relsr = []
    daysum = []

    for group in tqdm(groupby):
        port_code = group[0]
        group_data = group[1]
        start_date = group_data.iloc[0]['NavDate']
        end_date = group_data.iloc[-1]['NavDate']

        #annual_abs_nav
        start_nav = group_data.iloc[0]['Nav']
        end_nav = group_data.iloc[-1]['Nav']
        abs_nav = end_nav / start_nav
        dif = len(group_data)
        # dif = pd.period_range(start_date,end_date,freq='D')
        annual_abs_nav = pow(abs_nav, N / dif) - 1
        # print(port_code)
        # print('绝对年化收益率为：%f'%annual_abs_nav)

        #annual_rel_nav
        start_csi = csi[csi['Date'] == start_date]['Close'].iloc[0]
        end_csi = csi[csi['Date'] == end_date]['Close'].iloc[0]
        ratio_csi = end_csi / start_csi
        # print(ratio_csi)
        annual_market_nav = pow(ratio_csi, N / dif) - 1
        # print('大盘年化收益率为：%f'%annual_market_nav)
        annual_rel_nav = annual_abs_nav - annual_market_nav
        # print('相对年化收益率为：%f'%annual_rel_nav)

        #annual_abs_std
        rise_ratio = group_data['Nav'].div(group_data['Nav'].shift(1))-1
        group_data['RiseRatio'] = rise_ratio
        abs_std = np.nanstd(group_data['RiseRatio'])
        annual_abs_std = sqrt(N) * abs_std
        # print('绝对年化风险为：%f'%annual_abs_std)

        #annual_rel_std
        group_data = group_data.merge(csi, left_on='NavDate', right_on='Date', how='left')
        #csi_rise = group_data.Close.diff()
        #group_data['CsiRise'] = csi_rise
        csi_rise_ratio = group_data['Close'].div(group_data['Close'].shift(1))-1
        group_data['CsiRiseRatio'] = csi_rise_ratio
        group_data['ActiveRatio'] = group_data['RiseRatio']-group_data['CsiRiseRatio']
        rel_std = np.nanstd(group_data['ActiveRatio'])
        annual_rel_std = sqrt(N) * rel_std
        # print('相对年化风险为：%f'%annual_rel_std

        #sharpe ratio
        rf = 0.03
        abs_sr = (annual_abs_nav - rf) / annual_abs_std
        # print('绝对年化夏普率为：%f'%abs_sr)
        rel_sr = annual_rel_nav / annual_rel_std
        # print('相对年化夏普率为：%f'%rel_sr)

        code.append(port_code)
        startday.append(start_date)
        endday.append(end_date)
        absnav.append(annual_abs_nav)
        relnav.append(annual_rel_nav)
        absstd.append(annual_abs_std)
        relstd.append(annual_rel_std)
        abssr.append(abs_sr)
        relsr.append(rel_sr)
        daysum.append(dif)

    result = pd.DataFrame()
    result["port_code"] = code
    result["start_date"] = startday
    result["end_date"] = endday
    result["annual_abs_return"] = absnav
    result["annual_rel_return"] = relnav
    result["annual_abs_std"] = absstd
    result["annual_rel_std"] = relstd
    result["abs_sr"] = abssr
    result["rel_sr"] = relsr
    result["day_sum"] = daysum

    result.to_csv(toaddress,index=False)
    return


# In[ ]:


#result_train = vindex(train_date,"E:/GraduationProject/result_train.csv")
#result_total = vindex('2017-06-30',"E:/GraduationProject/result_total.csv")


# In[ ]:


daily = pd.read_csv('E:/GraduationProject/result_daily.csv')


# In[ ]:


def dailyprocessing(df,enddate):#对每日数据的处理
    '''
    df 选取的大V列表
    enddate 训练集的最后一天
    '''
    daily1 = daily[daily.NavDate>enddate]
    daily2 = daily1.merge(df,on='PortCode',how='right')
    daily2.dropna(axis=0,thresh=None, subset=["PortCode"], inplace=True)
    del daily2['Nav']
    daily2.reset_index(drop = True)
    return(daily2)


# In[ ]:


cal = pd.read_csv("E:/GraduationProject/result_total.csv")


# In[ ]:


rel_sr = pd.read_csv('E:/GraduationProject/excess_baseline_60.csv')


# In[ ]:


def delta2int(delta):
    timeint = delta.days
    return(timeint)


# In[ ]:


def choose_top_v(enddate,topn,n,m):
    '''
    enddate 计算指标选取大V的最后一天
    topn 选取几个人
    n 截止到计算指标大V最短持仓时间
    m 大V在训练集中持续时间
    '''
    cal1 = cal[cal.end_date > enddate]
    start_date = pd.to_datetime(cal1['start_date'])
    end_date = pd.to_datetime(cal1['end_date'])
    enddate_time = pd.to_datetime(enddate)
    cal1['diff'] = enddate_time-start_date
    cal1['testday'] = end_date-enddate_time
    test_data1 = cal1[cal1['diff'].apply(delta2int)>n]
    test_data = test_data1[test_data1['testday'].apply(delta2int)>m]
    rel_sr1 = rel_sr[rel_sr.NavDate==enddate]
    rel_sr2 = test_data.merge(rel_sr1,left_on='port_code',right_on='PortCode',how='left')
    df1 = rel_sr2.loc[:,['PortCode', 'sixty_sr']]
    df2 = df1.sort_values(by='sixty_sr',ascending=False)[0:topn].reset_index(drop = True)
    del df2['sixty_sr']
    return(df2)#df2返回的是portcode列表


# In[ ]:


#vindex('2017-06-30',"E:/GraduationProject/totalday_index.csv")


# In[ ]:


enddate_index = csi[csi['Date'] > train_date].index.values[0]-1 #训练集最后一天的index

N = 250 #交易日天数
t = 20 #滚动窗口天数
n = 15 #滚动窗口次数
nt = n*t
money = 1 #初始分配资金
member = 50
m = money/member #每个人初始金额
nav = m

changedate = []
navlist = []

for p in tqdm(range(n)):#用p来循环滚动窗口
    nav = m
    enddate = csi['Date'][enddate_index].strftime("%Y-%m-%d")#训练集最后一天
    print(enddate)
    memberlist = choose_top_v(enddate,member,200,60)
    daily2 = dailyprocessing(memberlist,enddate)
    #test = daily2[daily2.isnull().values==True]
    groupby = daily2.groupby('PortCode')#daily2包括测试集每天的收益率和大盘收益率
    navframe = pd.DataFrame()
    for group in groupby:#循环大V
        portcode = group[0]
        code_data = group[1].reset_index(drop = True)
        last_day = pd.to_datetime(code_data['NavDate'].iloc[-1])#该大V持仓最后一天
        end_day = pd.to_datetime('2017-06-30')
        length = len(code_data)
        if last_day<end_day:
            nowcsi = csi[csi.Date > last_day].reset_index(drop = True)
            for k in range(len(nowcsi)):#将天数不足的大V天数补齐，日收益率为0  
                str_date = nowcsi['Date'][k].strftime("%Y-%m-%d")
                code_data.loc[length+k]={'PortCode':portcode,'NavDate':str_date,'RiseRatio':0,'CsiRiseRatio':0}
        a = code_data['RiseRatio'][0:t]
        b = code_data['NavDate'][0:t]
        for i in range(t):#用i来循环每一次滚动窗口中t天，计算每天的净值
            a.iloc[i] = nav*(1+code_data['RiseRatio'].iloc[i])
            nav = a.iloc[i]    
            b.iloc[i] = code_data['NavDate'].iloc[i]  
        nav = m
        navframe['Date'] = b
        navframe[portcode] = a

    xdate = navframe['Date']
    del navframe['Date']
    navframe['Col_sum'] = navframe.apply(lambda x: x.sum(), axis=1)
    changedate.extend(list(xdate))
    navlist.extend(list(navframe['Col_sum']))
    nav_sum = navlist[-1]       
    m = nav_sum/member
    enddate_index = enddate_index+t


# In[ ]:


newnav = pd.DataFrame()
newnav["NavDate"] = changedate
newnav["Nav"] = navlist
newnav.to_csv("E:/GraduationProject/baseline.csv",index=False)


# In[ ]:


import dateutil,pylab  
from pylab import * 
def NavCurve(fileaddress):
    df = pd.read_csv(fileaddress,parse_dates=['NavDate'])
    daytime = list(df["NavDate"])
    navlist = list(df["Nav"])
    pylab.plot_date(pylab.date2num(daytime), navlist, marker='.', mfc='darkblue',linestyle='-')
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel("date") #X轴标签
    plt.ylabel("nav") #Y轴标签
    plt.title("baseline nav curve") #标题
    plt.show()


# In[ ]:


NavCurve("E:/GraduationProject/baseline.csv")


# In[ ]:


def portfolioindex(fileaddress):
    df = pd.read_csv(fileaddress, parse_dates=['NavDate'])
    csi = pd.read_csv('E:/GraduationProject/Xueqiu/csi300.csv', parse_dates=['Date'])

    start_date = df.iloc[0]['NavDate']
    end_date = df.iloc[-1]['NavDate']
    
    #annual_abs_nav
    start_nav = df.iloc[0]['Nav']
    end_nav = df.iloc[-1]['Nav']
    abs_nav = end_nav / start_nav
    dif = len(df)
    annual_abs_nav = pow(abs_nav, N / dif) - 1

    #annual_rel_nav
    start_csi = csi[csi['Date'] == start_date]['Close'].iloc[0]
    end_csi = csi[csi['Date'] == end_date]['Close'].iloc[0]
    ratio_csi = end_csi / start_csi
    annual_market_nav = pow(ratio_csi, N / dif) - 1
    annual_rel_nav = annual_abs_nav - annual_market_nav

    #annual_abs_std
    rise_ratio = [df['Nav'].iloc[i + 1] / df['Nav'].iloc[i]-1 for i in range(len(df) - 1)]
    rise_ratio.insert(0,float("nan"))
    df['RiseRatio'] = rise_ratio
    abs_std = np.nanstd(df['RiseRatio'])
    annual_abs_std = sqrt(N) * abs_std

    #annual_rel_std
    group_data = df.merge(csi, left_on='NavDate', right_on='Date', how='left')
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

    return


# In[ ]:


portfolioindex('E:/GraduationProject/baseline.csv')


# In[ ]:


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
start,end,maxdd,fill,length = GetMaxDrawdown(newnav,'Nav','NavDate')

