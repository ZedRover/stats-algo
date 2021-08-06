"""
统计软件与算法
2020年11月18日
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
os.chdir('C:/Users/raona/Desktop');     #更改路径


#linear regression
np.random.seed(1010)
x  = np.random.normal(1,1,30)
y = 20+x*2+np.random.uniform(10,15,30)


stats.linregress(x,y)

np.linalg.lstsq(np.vstack( [x,np.ones_like(x)] ).T,y)


########Example 11.4.2
x = np.arange(100)
y = 150 + 3*x + 0.03*x**2 + 5*np.random.randn(len(x))


# Create the Design Matrices
M1 = np.vstack( (np.ones_like(x), x) ).T
M2 = np.vstack( (np.ones_like(x), x, x**2) ).T
M3 = np.vstack( (np.ones_like(x), x, x**2, x**3) ).T
    
# Solve the equations
p1 = np.linalg.lstsq(M1, y)
p2 = np.linalg.lstsq(M2, y)
p3 = np.linalg.lstsq(M3, y)



import statsmodels.api as sm

Res1 = sm.OLS(y, M1).fit()
Res2 = sm.OLS(y, M2).fit()
Res3 = sm.OLS(y, M3).fit()
    
print(Res1.summary())
print(Res2.summary())
print(Res3.summary())

Res1.aic
Res1.params

'''Formula-based modeling, using the tools from statsmodels
    Input: x/y data
    '''

import statsmodels.formula.api as smf
    
# Turn the data into a pandas DataFrame,
 # so that we can address them in the formulas with their name
df = pd.DataFrame({'x':x, 'y':y})
    
# Fit the models, and show the results
Res1F = smf.ols('y~x', df).fit()
Res2F = smf.ols('y ~ x+I(x**2)', df).fit()
Res3F = smf.ols('y ~ x+I(x**2)+I(x**3)', df).fit()
    
print(Res1F.summary())
print(Res2F.summary())
print(Res3F.summary())


Res2F.params
Res2F.conf_int()
Res2F.rsquared





