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

############LOGISTIC REGRESSION
def logistic(x, beta, alpha=0):
    ''' Logistic Function '''
    return 1.0 / (1.0 + np.exp(np.dot(beta, x) + alpha))

##Page 186 Example
# additional packages
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Binomial

# Get the data
inFile = 'challenger_data.csv'
challenger_data = np.genfromtxt(inFile, skip_header=1,
                    usecols=[1, 2], missing_values='NA',
                    delimiter=',')
# Eliminate NaNs
challenger_data = challenger_data[~np.isnan(challenger_data[:, 1])]

# Create a dataframe, with suitable columns for the fit
df = pd.DataFrame()
df['temp'] = np.unique(challenger_data[:,0])
df['failed'] = 0
df['ok'] = 0
df['total'] = 0
df.index = df.temp.values

# Count the number of starts and failures
for ii in range(challenger_data.shape[0]):
    curTemp = challenger_data[ii,0]
    curVal  = challenger_data[ii,1]
    df.loc[curTemp,'total'] += 1
    if curVal == 1:
        df.loc[curTemp, 'failed'] += 1
    else:
        df.loc[curTemp, 'ok'] += 1

# fit the model

# --- >>> START stats <<< ---
model = glm('ok + failed ~ temp', data=df, family=Binomial()).fit()
# --- >>> STOP stats <<< ---

print(model.summary())





#Data split
from sklearn.model_selection import train_test_split

X, y = np.arange(10).reshape((5, 2)), range(5)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
print(X_train)
print(X_test)
print(y_train)
print(y_test)


from sklearn.linear_model import LinearRegression
########Example 11.4.2
X = np.arange(100).reshape(-1,1)
y = 150 + 3*X + 0.03*X**2 + 5*np.random.randn(len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

reg_all = LinearRegression()
reg_all.fit(X_train,y_train)

y_pred = reg_all.predict(X_test)
reg_all.score(X_test,y_test)

#Cross validation
from sklearn.model_selection import cross_val_score

cv_results = cross_val_score(reg,X,y,cv = 5,scoring = 'neg_mean_squared_error')
print(cv_results)

np.mean(cv_results)



##Bootstrap
sample = ['A','B','C','D','E']

bootstrap_sample = np.random.choice(sample,size=4,replace = True)
print(bootstrap_sample)

bootstrap_sample1 = np.random.choice(sample,size=4,replace = False)
print(bootstrap_sample1)









