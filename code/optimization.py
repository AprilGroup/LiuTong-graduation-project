import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import time
import dateutil,pylab  
from pylab import * 
#%%

N=250
train_date = '2015-12-31'

def vindex(df,toaddress,col_group,col_nav):#计算在坑内时大V指标

    groupby = df.groupby(col_group)

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

    for group in groupby:
        port_code = group[0]
        group_data = group[1]
        start_date = group_data.iloc[0]['NavDate']
        end_date = group_data.iloc[-1]['NavDate']

        #annual_abs_nav
        start_nav = group_data.iloc[0][col_nav]
        end_nav = group_data.iloc[-1][col_nav]
        abs_nav = end_nav / start_nav
        dif = len(group_data)
        # dif = pd.period_range(start_date,end_date,freq='D')
        annual_abs_nav = pow(abs_nav, N / dif) - 1
        # print(port_code)
        # print('绝对年化收益率为：%f'%annual_abs_nav)

        #annual_rel_nav
        start_csi = hs300[hs300['Date'] == start_date]['Close'].iloc[0]
        end_csi = hs300[hs300['Date'] == end_date]['Close'].iloc[0]
        ratio_csi = end_csi / start_csi
        # print(ratio_csi)
        annual_market_nav = pow(ratio_csi, N / dif) - 1
        # print('大盘年化收益率为：%f'%annual_market_nav)
        annual_rel_nav = annual_abs_nav - annual_market_nav
        # print('相对年化收益率为：%f'%annual_rel_nav)

        #annual_abs_std
        abs_std = np.nanstd(group_data['RiseRatio'])
        annual_abs_std = sqrt(N) * abs_std
        # print('绝对年化风险为：%f'%annual_abs_std)

        #annual_rel_std
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
    return result
#%%
hs300 = pd.read_csv('E:/GraduationProject/hs300.csv', parse_dates=['Date'])

def CloseCurve(df):#做折线图
    daytime = list(df["Date"])
    closelist = list(df["Close"])
    plt.figure(figsize=(8,5))
    pylab.plot_date(pylab.date2num(daytime), closelist, color='b',alpha=0.8,marker='',linestyle='-')

    plt.subplots_adjust(bottom=0.2)

    plt.xlabel("Date") #X轴标签
    plt.ylabel("Close") #Y轴标签
    plt.grid(which= 'major')
    plt.legend() 
    plt.title("hs300 Close Curve") #标题
    plt.show()
#%%
CloseCurve(hs300)
hs300_train = hs300[hs300.Date<=train_date]
#%%
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
start,end,maxdd,fill,length = GetMaxDrawdown(hs300_train,'Close','Date')
#%%
position = pd.read_csv('E:/GraduationProject/position_frame.csv', parse_dates=['date'])
maxdrawdown = pd.read_csv('E:/GraduationProject/maxdrawdown.csv')

#截取出现坑的起始时间
start_date = maxdrawdown['start'].iloc[0]
end_date = maxdrawdown['end'].iloc[0]

position1 = position[position.date>start_date]
position_use = position1[position1.date<end_date]
#%%
#根据历史平均仓位和出现坑时的平均仓位之差选大V
FallPeriodPosition = position_use.groupby(by=['vcode'], as_index=False)['position'].mean()
TotalPosition = position.groupby(by=['vcode'], as_index=False)['position'].mean()
MeanPositionByVcode = FallPeriodPosition.merge(TotalPosition,on='vcode',how='left')
MeanPositionByVcode.rename(columns={"position_x": "FallPeriodPosition", "position_y": "TotalPosition"},inplace=True)
MeanPositionByVcode['diff'] = MeanPositionByVcode['TotalPosition']-MeanPositionByVcode['FallPeriodPosition']
topn = int(len(MeanPositionByVcode)/10)
v_position_choose = MeanPositionByVcode.sort_values(by='diff',ascending=False)[0:topn].reset_index(drop=True)
vcode_position_choose = set(v_position_choose['vcode'])
#%%
#根据坑内时段夏普率挑选大V
result_daily = pd.read_csv('E:/GraduationProject/result_daily.csv', parse_dates=['NavDate'])
result_daily1 = result_daily[result_daily.NavDate>start_date]
result_daily_use = result_daily1[result_daily1.NavDate<end_date].reset_index(drop=True)

#result = vindex(result_daily_use,'E:/GraduationProject/result_vindex.csv','PortCode','Nav')
result = pd.read_csv('E:/GraduationProject/result_vindex.csv')
result1 = result[result.day_sum>50]
topn1 = int(len(result1)/10)
v_relsr_choose = result1.sort_values(by='rel_sr',ascending=False)[0:topn1].reset_index(drop=True)
vcode_relsr_choose = set(v_relsr_choose['port_code'])
#%%
#选取爬坑速度快的大V
'''
vportcode = []
drawdownlist = []
startlist =[]
endlist = []
increaselist = []

for v in tqdm(list(set(result_daily_use['PortCode']))):
    vdata = result_daily_use[result_daily_use['PortCode'] == v].reset_index(drop=True)
    if len(vdata)<50:
        continue
    else:
        start_date,end_date,maxdd,fill_date,fill_length = GetMaxDrawdown(vdata,'Nav','NavDate')
        last_index = vdata.index[-1]
        low_index = vdata[vdata.NavDate == end_date].index[0]
        if low_index>last_index-30:
            continue
        else:
            low_nav = vdata['Nav'].loc[low_index]
            check_index = low_index+30
            check_nav = vdata['Nav'].loc[check_index]
            increase = (check_nav-low_nav)/low_nav
            vportcode.append(v)
            drawdownlist.append(maxdd)
            startlist.append(start_date)
            endlist.append(end_date)
            increaselist.append(increase)

maxdd_frame = pd.DataFrame()
maxdd_frame['PortCode'] = vportcode
maxdd_frame['Start'] = startlist
maxdd_frame['End'] = endlist
maxdd_frame['Maxdd'] = drawdownlist
maxdd_frame['increase'] = increaselist

topn2 = int(len(maxdd_frame)/10)
v_maxdd_choose = maxdd_frame.sort_values(by='increase',ascending=False)[0:topn2].reset_index(drop=True)
maxdd_frame.to_csv('E:/GraduationProject/maxdd_frame.csv')
vcode_maxdd_choose = set(v_maxdd_choose['PortCode'])
'''

maxdd_frame = pd.read_csv('E:/GraduationProject/maxdd_frame.csv')
topn2 = int(len(maxdd_frame)/10)
v_maxdd_choose = maxdd_frame.sort_values(by='increase',ascending=False)[0:topn2].reset_index(drop=True)
vcode_maxdd_choose = set(v_maxdd_choose['PortCode'])
#%%
#根据总体相对夏普率挑选大V
result_total = pd.read_csv("E:/GraduationProject/result_total.csv")
result_train = pd.read_csv("E:/GraduationProject/result_train.csv")
rel_sr = pd.read_csv('E:/GraduationProject/thirtydays_sr1.csv')
excess_result = pd.read_csv('E:/GraduationProject/excess_result.csv')
result_200 = pd.read_csv('E:/GraduationProject/200_result.csv')
#用total的relsr挑选效果更好 尝试在7月之后用整体rel_sr挑选或test_rel_sr挑选

def delta2int(delta):
    timeint = delta.days
    return(timeint)
    
def choose_top_v1(enddate,topn,n,m):
    '''
    enddate 计算指标选取大V的最后一天
    topn 选取几个人 300
    n 截止到计算指标 大V持仓时间阈值 250
    m 大V在训练集中持续时间 50
    '''
    cal1 = result_total[result_total.end_date > enddate]
    calvcode = pd.DataFrame(cal1['port_code'])
    start_date = pd.to_datetime(cal1['start_date'])
    end_date = pd.to_datetime(cal1['end_date'])
    enddate_time = pd.to_datetime(enddate)
    cal1['diff'] = enddate_time-start_date#有问题计算的不是工作日天数
    cal1['testday'] = end_date-enddate_time
    test_data1 = cal1[cal1['diff'].apply(delta2int)>n]
    test_data = test_data1[test_data1['testday'].apply(delta2int)>m]
    rel_sr1 = test_data.merge(rel_sr,left_on='port_code',right_on='PortCode',how='left') 
    df1 = rel_sr1.loc[:,['PortCode', enddate]]
    df2 = df1.sort_values(by=[enddate],ascending=False)[0:topn].reset_index(drop = True)
    del df2[enddate] 
    vcode_total_choose = set(df2['PortCode'])#返回的df2是选取的大V的portcode列表 set
    #结合所有大V一起
    set1 = vcode_position_choose.union(vcode_relsr_choose)
    set2 = set1.union(vcode_maxdd_choose)
    vcode_choose = list(set2.union(vcode_total_choose))
    vcode_frame=pd.DataFrame({'PortCode':vcode_choose})

    #根据超额收益计算的夏普率选取100名大V 
    excess_result1 = excess_result.merge(calvcode,left_on='PortCode',right_on='port_code',how='right')
    del excess_result1['port_code']
    excess_result2 = excess_result1[excess_result1.NavDate == enddate]
    excess_result3 = excess_result2.merge(vcode_frame,on='PortCode',how='inner')
    excess_result4 = excess_result3.merge(result_train,left_on='PortCode',right_on='port_code',how='inner')
    excess_result5 = excess_result4[excess_result4.day_sum>200]
    excess_result6 = excess_result5.loc[:,['port_code', 'annul_thirty_risk','thirty_sr']]
    
    excess_choose = excess_result6.sort_values(by='annul_thirty_risk')[0:200].reset_index(drop=True)    
    excess_choose1 = excess_choose.sort_values(by='thirty_sr',ascending=False)[0:100].reset_index(drop=True)   

    excess_v_code = excess_choose1['port_code']
    
    #构成100名大V的收益率矩阵
    return_frame = pd.DataFrame()
    relsr_list = []
    for v in list(set(excess_v_code)):
        daily_data = result_daily[result_daily.PortCode == v].reset_index(drop=True)
        daily_data1 = daily_data[daily_data.NavDate<=enddate]
        #daily_data1['ARR'] = daily_data1['RiseRatio']-daily_data1['CsiRiseRatio']
        result_2001 = result_200[result_200.NavDate<=enddate]
        last_index = daily_data1.index[-1]
        start_index = last_index-200
        start_day = daily_data['NavDate'].loc[start_index]
        v_return = list(daily_data1[daily_data1.NavDate>start_day]['RiseRatio'])
        return_frame[v] = v_return
        v_relsr = result_2001[result_2001.PortCode == v]['200_sr'].iloc[-1]
        relsr_list.append(v_relsr)
    #matrix = np.matrix(return_frame.as_matrix())
    #print(np.isnan(matrix).sum())
    
    #import seaborn as sns
    #import matplotlib.pyplot as plt
    
    dfData = return_frame.corr()
    #plt.subplots(figsize=(9, 9)) # 设置画面大小
    #sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap="Blues")
    #plt.savefig('E:/GraduationProject/BluesStateRelation.png')
    #plt.show()
    
    corr_matrix = np.matrix(dfData.as_matrix())
    relsr_matrix = np.matrix(relsr_list)#相对夏普率行矩阵
    relsr_matrix_t = relsr_matrix.T#相对夏普率列矩阵
    corr_matrix_i = corr_matrix.I 
    #计算权重

    h = (corr_matrix_i*relsr_matrix_t)/(relsr_matrix*corr_matrix_i*relsr_matrix_t)
    h_t = h.T
    h_list = h_t.tolist()[0]

    
    weight = pd.DataFrame({'PortCode':excess_v_code,'Weight':h_list})
    weight_plus = weight[weight.Weight>0].reset_index(drop=True)
    if len(weight_plus)>=50:
        weight_choose = weight_plus.sort_values(by='Weight',ascending=False)[0:50].reset_index(drop=True)
        weight_choose_sum = weight_choose['Weight'].sum()
        weight_choose['AdjWeight'] = weight_choose['Weight']/weight_choose_sum
    else:
        weight_choose = weight_plus.sort_values(by='Weight',ascending=False).reset_index(drop=True)
        weight_choose_sum = weight_choose['Weight'].sum()
        weight_choose['AdjWeight'] = weight_choose['Weight']/weight_choose_sum

    daily1 = result_daily[result_daily.NavDate>enddate]
    daily2 = daily1.merge(weight_choose,on='PortCode',how='right')
    daily2.dropna(axis=0,thresh=None, subset=["PortCode"], inplace=True)
    daily2.dropna()
    del daily2['Nav']
    del daily2['Weight']
    del daily2['AdjWeight']
    daily3 = daily2.reset_index(drop = True) 
    return(weight_choose,daily3)

#%%
#轮动窗口实验
enddate_index = hs300[hs300['Date'] > train_date].index.values[0]-1 #训练集最后一天的index

N = 250 #交易日天数
t = 20 #滚动窗口天数
n = 15 #滚动窗口次数
nt = n*t

nav_sum = 1 #初始分配资金

changedate = []
navlist = []

v_frame = pd.DataFrame()

for p in tqdm(range(n)):#用p来循环滚动窗口
    enddate = hs300['Date'][enddate_index].strftime("%Y-%m-%d")#训练集最后一天
    print(enddate)
    memberweight,daily = choose_top_v1(enddate,500,250,50)
    #test = daily[daily.isnull().values==True]
    print(memberweight)
    v_list = list(memberweight['PortCode'][0:30])
    v_frame[str(p)]=v_list
    #daily包括测试集每天的收益率和大盘收益率
    navframe = pd.DataFrame()
    for v in list(set(memberweight['PortCode'])):#循环大V
        weight_data = memberweight[memberweight.PortCode == v].reset_index(drop = True)
        code_data = daily[daily.PortCode == v].reset_index(drop = True)
        last_day = pd.to_datetime(code_data['NavDate'].iloc[-1])#该大V持仓最后一天
        end_day = pd.to_datetime('2017-06-30')
        length = len(code_data)
        if last_day<end_day:
            nowcsi = hs300[hs300.Date > last_day].reset_index(drop = True)
            for k in range(len(nowcsi)):#将天数不足的大V天数补齐，日收益率为0  
                str_date = nowcsi['Date'][k].strftime("%Y-%m-%d")
                code_data.loc[length+k]={'PortCode':v,'NavDate':str_date,'RiseRatio':0,'CsiRiseRatio':0}
        a = list(code_data['RiseRatio'][0:t])
        b = list(code_data['NavDate'][0:t])
        nav = nav_sum*weight_data['AdjWeight'].iloc[0]
        for i in range(t):#用i来循环每一次滚动窗口中t天，计算每天的净值
            a[i] = nav*(1+code_data['RiseRatio'].iloc[i])
            nav = a[i]    
            b[i] = code_data['NavDate'].iloc[i]  
        navframe['Date'] = b
        navframe[v] = a

    xdate = navframe['Date']
    del navframe['Date']
    navframe['Col_sum'] = navframe.apply(lambda x: x.sum(), axis=1)
    changedate.extend(list(xdate))
    navlist.extend(list(navframe['Col_sum']))
    nav_sum = navlist[-1]
       
    enddate_index = enddate_index+t
   
v_frame.to_csv("E:/GraduationProject/30days_v.csv",index=False)
#%%
newnav = pd.DataFrame()
newnav["NavDate"] = changedate
newnav["Nav"] = navlist
newnav.to_csv("E:/GraduationProject/adjweight_30.csv",index=False)

from pylab import * 
def NavCurve(fileaddress):
    df = pd.read_csv(fileaddress,parse_dates=['NavDate'])
    daytime = list(df["NavDate"])
    navlist = list(df["Nav"])
    plt.figure(figsize=(8,5))
    pylab.plot_date(pylab.date2num(daytime), navlist, color='b',alpha=0.8,marker='',linestyle='-')

    plt.subplots_adjust(bottom=0.2)

    plt.xlabel("Date") #X轴标签
    plt.ylabel("Nav") #Y轴标签
    plt.grid(which= 'major') 
    plt.title("Nav Curve") #标题
    plt.show()
NavCurve("E:/GraduationProject/adjweight_30.csv")

#%%
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

#%%
portfolioindex('E:/GraduationProject/adjweight_30.csv')

