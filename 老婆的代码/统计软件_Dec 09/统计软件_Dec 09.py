"""
统计软件与算法
2020年12月9日
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
    #更改路径
os.chdir("/Users/zed/VSCode/Stats-Alg/老婆的代码/统计软件_Dec 09")

"""第五章 时间序列"""
"""5.1时间序列的图形描述"""
#%matplotlib inline
from matplotlib.pylab import rcParams 
rcParams['figure.figsize'] = 15, 6

##画图
df= pd.read_csv('Airports.csv')
df.index=pd.DatetimeIndex(freq='M',start='1995-01',periods=108)
plt.plot(df)
plt.xlabel('Time')
plt.ylabel('Number of passengers')
plt.title('Number of passengers in 12 airports of China')
plt.legend(labels=df.columns,loc=2)

#局部图像
plt.plot(df[(8*12):])
plt.xlabel('Time')
plt.ylabel('Number of passengers')
plt.title('Number of passengers in 12 airports of China')
plt.legend(labels=df.columns,loc=1)


#定义滑动平均函数
def mplot(ts,loc=1,tit=''):
    ts.dropna(inplace=True)
    RB=pd.Series.rolling(ts,window=12).mean()
    RS=pd.Series.rolling(ts,window=12).std()
    plt.plot(ts,label='The series')
    plt.plot(RB,label='The rolling mean')
    plt.plot(RS,label='The rolling std')
    plt.title(tit)
    plt.legend(loc=loc)
    
BJ=df['Beijing'][:(8*12)]
BJ.index=pd.DatetimeIndex(freq='M',start='1995-01',periods=8*12)
mplot(BJ,2,'Number of Passengers in Beijing Airport')


df = pd.DataFrame({'B': np.arange(10)})
print(df.rolling(5, center=True).mean())
print(df.rolling(5).mean())


"""5.2时间序列平稳性"""
#ADF test
from statsmodels.tsa.stattools import adfuller
def adf_test(ts):
    ts.dropna(inplace=True)
    stat,pvalue,N_lags,N_obs=adfuller(ts,autolag='AIC')[0:4]
    print('Test Stat={}, p-value={}, #lags={},\
    #obs={}'.format(stat,pvalue,N_lags,N_obs))
      
      
##EXAMPLE 5.2
darwin=np.loadtxt('darwin.txt')
dates=pd.date_range(start='1882-01',periods=len(darwin), freq='M')
Darwin=pd.DataFrame(darwin,columns=['SLP'])
Darwin['date']=dates
D=pd.Series(np.array(Darwin['SLP']),index=Darwin['date'])

mplot(D,2,'Darwin Sea Level Pressure')

##5.2.2减去滑动平均的尝试
BJ_rm=pd.Series.rolling(BJ,12).mean()
BJD=BJ-BJ_rm
mplot(BJD,2,'Number of Passengers in Beijing Airport')

adf_test(BJD)

##5.2.3序列的差分
BJ_diff=BJ-BJ.shift()
BJ_rm=pd.Series.rolling(BJ_diff,12).mean()
BJD=BJ_diff-BJ_rm
mplot(BJD,3,'Number of Passengers in Beijing Airport')
adf_test(BJD)

##5.2.4序列的季节差分
from statsmodels.tsa.seasonal import seasonal_decompose
s=seasonal_decompose(BJ)
plt.plot(BJ,label='original BJ')
plt.plot(s.trend,label='trend')
plt.plot(s.seasonal,label='seasonal')
plt.plot(s.resid,label='residual')
plt.legend(loc=2)
plt.title('Decomposition of time series')
adf_test(s.resid)


"""5.3 ARMA模型的拟合和预测"""
from statsmodels.tsa.stattools import acf, pacf

def ACF_PACF(ts,lag=20):
    lag_acf = acf(ts, nlags=lag)
    lag_pacf = pacf(ts, nlags=lag, method='ols')
    #画 ACF: 
    plt.subplot(121) 
    plt.vlines(range(lag),[0],lag_acf,linewidth=5.0)
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle=':',color='blue')
    plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='red')
    plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='red')
    plt.title('Autocorrelation Function')
    #画 PACF:
    plt.subplot(122)
    plt.vlines(range(lag),[0],lag_pacf,linewidth=5.0)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle=':',color='blue')
    plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='red')
    plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='red')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()

ACF_PACF(BJ)
ACF_PACF(s.resid)

"""5.4 新西兰奥克兰降水数据的ARMA拟合"""
A=pd.read_csv('NZRainfall.csv')
A=np.array(A['Auckland'])
dat=pd.date_range(start='2000-01',periods=len(A), freq='M')
NZA=pd.DataFrame(A,columns=['A_Rain'])
NZA['date']=dat
NZA=pd.Series(np.array(NZA['A_Rain']),index=NZA['date'])
plt.plot(NZA)
plt.title('Rain Fall of Auckland of Zealand')

ACF_PACF(NZA)


##ARMA(6,0)
from statsmodels.tsa.arima_model import ARMA
A_60=ARMA(NZA,(6,0)).fit()
print(A_60.params)


##残差正态性检验
stats.normaltest(A_60.resid)

##Ljung-Box检验
import statsmodels.api as sm
r,q,p=sm.tsa.acf(A_60.resid.values.squeeze(),qstat=True)
dt=np.c_[range(1,len(q)+1),r[1:],q,p]
table=pd.DataFrame(dt,columns=['lag','AC','Q','Prob(>Q)'])
print(table.set_index('lag'))
plt.plot(p,'o')
plt.axhline(y=0,linestyle=':',color='blue')
plt.xlabel('lag')
plt.ylabel('p-value')
plt.title("Ljung-Box Test's p-value" )

#预测
pred_A=A_60.predict('2011-10-31','2012-10-31')
plt.plot(NZA['2009-01-31':],label='NZA')
plt.plot(pred_A,'r--',label='Prediction',linewidth=5)
plt.legend(loc='best')

"""5.5向量自回归模型"""
##5.51 VAR
S= pd.read_csv('soi.csv')
S.index=pd.DatetimeIndex(freq='M',start='1992-02',periods=len(S))
SS=np.log(S).diff().dropna()


from statsmodels.tsa.api import VAR
model=VAR(SS)
res=model.fit(maxlags=20,ic='bic') #根据BIC自动筛选
res.summary()

lag_order=res.k_ar #= order 3
print (res.forecast(SS.values[-lag_order:],20)) #预测20个      
res.plot_forecast(20)

res.plot_sample_acorr()


##5.5.2 Granger因果检验
res.test_causality('VS',['NO','TI'],kind='f') #Granger因果检验


##5.5.3脉冲响应分析
irf = res.irf(10) #看10步的响应
irf.plot(orth=False)
irf.plot_cum_effects(orth=False)
fevd=res.fevd(5)
fevd.summary()
res.fevd(20).plot()