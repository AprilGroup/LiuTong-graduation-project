
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from math import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import time


# In[2]:


#df = pd.read_csv('E:/GraduationProject/Xueqiu/records.csv')
#df['Updated'] = df['Updated'].apply(lambda x:str(x)[0:10])


# In[3]:


record = pd.read_csv('E:/GraduationProject/records_use.csv', parse_dates=['Updated'])
record1 = record[record.Updated<'2017-07-01']
record2 = record1[record1.Updated != '2015-01-01']
record3 = record2[record2.Updated != '2015-01-02']
record4 = record3[record3.Updated != '2015-09-03']
record = record4[record4.Updated != '2015-09-04']


# In[4]:


quote = pd.read_csv('E:/GraduationProject/quote_use.csv', parse_dates=['TradingDay'])
quote = quote[quote.TradingDay<'2017-07-01']
total_stock = list(set(quote['SecuCode']))


# In[5]:


hs300 = pd.read_csv('E:/GraduationProject/hs300.csv', parse_dates=['Date'])


# In[6]:


#def dateformat(cur_date):
    #time_format = cur_date.strftime("%Y-%m-%d") 
    #return(time_format)


# In[7]:


drop_v_list = []


# In[8]:


def cal_today_position(tmp_pos):
    today_pos = (tmp_pos/total_today_pos)*100
    return(today_pos)


# In[9]:


#new = pd.DataFrame({'PortCode': [], 'Updated':[], 'SecuCode':[],})
#new.join(vdata, on = '')
position_frame = pd.DataFrame()
vcode = []
update_day = []
position_sum = []

for v in tqdm(list(set(record['PortCode']))):
    print(v)
    vdata = record[record['PortCode'] == v].reset_index(drop=True)
    print(vdata)
    update = list(set(vdata['Updated']))#所有更新的日期
    stockcode = list(set(vdata['SecuCode']))#所有买过的股票代码
    if set(stockcode).issubset(set(total_stock)):#如果所有买过的股票是quote中有的股票的子集
        start_date = vdata['Updated'].iloc[0]
        end_date = vdata['Updated'].iloc[-1]
        print(start_date)
        print(end_date)
        hs300_1 = hs300[hs300.Date>=start_date]
        use_day = hs300_1[hs300_1.Date<=end_date].reset_index(drop=True)                  
        vmerge = use_day.merge(vdata,left_on='Date',right_on='Updated',how='left')
        vtable1 = vmerge.drop(['Close','PortCode','Updated','SecuCode','StockName','PrevWeight','TargetWeight'],axis=1)
        vtable = vtable1.drop_duplicates() 
        table_length = len(vtable)
        for stock in stockcode:
            vtable[str(stock)] = [0]*table_length
        vtable['cash'] = [0]*table_length
        vtable['position_sum'] = [0]*table_length   
        days = vtable['Date']
        vtable = vtable.set_index('Date')
        
        for this_date in list(days):#有换仓记录的从第一天到最后一天的所有工作日        
            if this_date in update:
                #换仓日的仓位计算方法
                vdata_tmp = vdata[vdata.Updated == this_date]#取record里这一天的换仓记录
                for j in list(set(vdata_tmp['SecuCode'])):#遍历当天换仓的所有股票代码
                    vdata_tmp1 = vdata_tmp[vdata_tmp.SecuCode == j]#取出当前遍历到的股票代码的那一条数据
                    vtable[str(j)].loc[this_date] = vdata_tmp1['TargetWeight'].iloc[0]
                v_tmp = vtable.loc[this_date]
                vtable['position_sum'].loc[this_date] = v_tmp.sum()
                vtable['cash'].loc[this_date] = 100-vtable['position_sum'].loc[this_date]
                if vtable['cash'].loc[this_date]<0:
                    vtable['cash'].loc[this_date] = 0
            else:
                #非换仓日时的仓位计算方法
                #取前一天的仓位数据乘以当天股票的涨跌幅除以总仓位得到新的仓位
                cal_tmp = days[days<this_date]
                yesterday = cal_tmp.iloc[-1]
                yesterday_position = vtable.loc[yesterday]

                new = pd.DataFrame()
                secucode = []
                todayposition = []#记录每次乘以涨跌幅之后的临时仓位 
                
                for stocki in stockcode:
                    yesterday_pos = yesterday_position[str(stocki)]
                    if yesterday_pos>0:
                        today_quote1 = quote[quote.TradingDay == this_date]
                        today_quote = today_quote1[today_quote1.SecuCode == stocki].reset_index(drop=True) 
                        if today_quote.empty:
                            pchange = 0
                        else:
                            pchange = today_quote['ChangePCT'].iloc[0]/100
                        today_tmp_pos = (1+pchange)*yesterday_pos#还不是最终结果，要除以总仓位
                        secucode.append(stocki)
                        todayposition.append(today_tmp_pos)                    
                    else:
                        today_tmp_pos = yesterday_pos
                        secucode.append(stocki)
                        todayposition.append(today_tmp_pos)

                new['secucode'] = secucode
                new['today_tmp_pos'] = todayposition
                total_today = new['today_tmp_pos'].sum()
                total_today_pos = total_today+vtable['cash'].loc[yesterday]
                new['today_pos'] = new['today_tmp_pos'].apply(cal_today_position)
                new = new.set_index('secucode')
                
                for stockj in stockcode:#更新vtable数据
                    vtable[str(stockj)].loc[this_date] = new['today_pos'].loc[stockj]
                v_tmp = vtable.loc[this_date]
                vtable['position_sum'].loc[this_date] = v_tmp.sum()
                vtable['cash'].loc[this_date] = 100-vtable['position_sum'].loc[this_date]   
                if vtable['cash'].loc[this_date]<0:
                    vtable['cash'].loc[this_date] = 0
        
        vtable = vtable.sort_index()
        print(vtable)

    else:
        drop_v_list.append(v)
        continue
    
    vtable_last = vtable.reset_index() 
    xvcode = [v]*len(vtable_last)
    xdate = vtable_last['Date']
    xposition = vtable_last['position_sum']
    
    vcode.extend(list(xvcode))
    update_day.extend(list(xdate))
    position_sum.extend(list(xposition))



# In[10]:


position_frame['vcode'] = vcode
position_frame['date'] = update_day
position_frame['position'] = position_sum


# In[11]:


position_frame.to_csv("E:/GraduationProject/position_frame.csv",index=False)

