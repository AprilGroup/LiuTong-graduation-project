
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


# In[4]:


rel_nav = pd.read_csv('E:/GraduationProject/result_daily.csv',parse_dates=['NavDate'])
rel_nav['ExcessRatio']=rel_nav['RiseRatio']-rel_nav['CsiRiseRatio']
excessnav = pd.DataFrame()
for v in tqdm(list(set(rel_nav['PortCode']))):
    vdata = rel_nav[rel_nav['PortCode'] == v].reset_index(drop=True)
    initnav = vdata['Nav'].iloc[0]
    relnav = list(vdata['Nav'])    
    for i in range(len(vdata)-1):
        relnav[i+1]=relnav[i]*(1+vdata['ExcessRatio'].iloc[i+1])
    vdata['RelNav']=relnav
    excessnav = pd.concat([excessnav,vdata],ignore_index=True)
excessnav[0:100]


# In[5]:


def cal_annul_return(x):
    y = pow(x,250/60)-1
    return y

def cal_annul_risk(x):
    y = x * pow(250,0.5)
    return y

excess = pd.DataFrame()
for v in tqdm(list(set(excessnav['PortCode']))):
    vdata = excessnav[excessnav['PortCode'] == v].reset_index(drop=True)
    if len(vdata)<60:
        continue
    vdata['sixty_return'] = vdata['RelNav'].div(vdata['RelNav'].shift(59))
    vdata['annul_sixty_return'] = vdata['sixty_return'].apply(cal_annul_return)
    risk = [float('nan')]*59
    for idx in range(59,len(vdata)):
        tmp = np.std(vdata['ExcessRatio'].iloc[idx-59:idx+1])
        risk.append(tmp)
    vdata['sixty_risk'] = risk
    vdata['annul_sixty_risk'] = vdata['sixty_risk'].apply(cal_annul_risk)
    vdata['sixty_sr']=(vdata['annul_sixty_return']-0.03)/vdata['annul_sixty_risk']
    excess = pd.concat([excess,vdata],ignore_index=True)

excess.to_csv('E:/GraduationProject/excess_baseline_60.csv',index=False)

