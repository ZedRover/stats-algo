"""
统计软件与算法
2020年10月21"""

"""
CHAPTER 6
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
os.chdir('C:/Users/raona/Desktop');     #更改路径

"""
Continuous RV
"""

#normal distribution
?stats.norm         #check stats.norm function
mu = -2
sigma = 0.7
nd = stats.norm(mu,sigma)

nd.rvs(size = 10)
stats.norm.rvs(mu,sigma,10)

level = 0.05
nd.ppf([level/2,1-level/2])


#t-distribution
n = 20
df = n-1
alpha = 0.05

print(stats.t(df).isf(alpha/2))
print(stats.t(df).ppf(1-alpha/2))      #both ppf and isf give same result

print(stats.norm.isf(alpha/2))

#CI
data = np.array([3.04,2.94,3.01,3.00,2.94,2.91,3.02,3.04,3.09,2.95,2.99,3.10,3.02])
alpha = 0.05
df = len(data)-1
mu = np.mean(data)
sigma = stats.sem(data)

(mu-sigma*stats.t(df).isf(alpha/2),mu+sigma*stats.t(df).isf(alpha/2))


?stats.t.interval
ci = stats.t.interval(1-alpha,df,loc = np.mean(data),scale = stats.sem(data))
ci

#chi-square distribution
?stats.chi2

data = np.array([3.04,2.94,3.01,3.00,2.94,2.91,3.02,3.04,3.09,2.95,2.99,3.10,3.02])

sigma = 0.05
chi2Dist = stats.chi2(len(data)-1)

statistic = sum(((data-np.mean(data))/sigma)**2)
statistic

chi2Dist.sf(statistic)


#F-distribution
method1 = np.array([20.7,20.3,20.3,20.3,20.7,19.9,19.9,19.9,20.3,20.3,19.7,20.3])
method2 = np.array([19.7,19.4,20.1,18.6,18.8,20.2,18.7,19])

fval = np.var(method1,ddof = 1)/np.var(method2,ddof=1)
fval

fd = stats.f(len(method1)-1,len(method2)-1)
p_oneTail = fd.cdf(fval)
p_oneTail

if(p_oneTail<0.025) or (p_oneTail>0.975):
    print("There is a significant difference between the two distributions.")
else:
    print("No significant difference.")

"""Other continous RV"""
#lognoraml
X = stats.lognorm.pdf(x = np.arange(0, 2.5, 0.01),s = 1)
plt.plot(X)
#Weibull distribution
'''Utility function to show Weibull distributions'''
    
t = np.arange(0, 2.5, 0.01)
lambdaVal = 1
ks = [0.5, 1, 1.5, 5]
for k in ks:
   wd = stats.weibull_min(k)
   plt.plot(t, wd.pdf(t), label='k = {0:.1f}'.format(k))
        
plt.xlim(0,2.5)
plt.ylim(0,2.5)
plt.xlabel('X')
plt.ylabel('pdf(X)')
plt.legend()

#exponential distribution
'''Utility function to show exponential distributions'''
?stats.expon

t = np.arange(0, 3, 0.01)
lambdas = [0.5, 1, 1.5]
for par in lambdas:
    plt.plot(t, stats.expon.pdf(t, 0, 1/par), label='$\lambda={:.1f}$'.format(par))
plt.legend()
        
plt.xlim(0,3)
plt.xlabel('X')
plt.ylabel('pdf(X)')
plt.axis('tight')
plt.legend()

###uniform dist
?stats.uniform

"""
Chapter7
"""

'''Check if the distribution is normal.'''  

n = 100
samples = stats.norm.rvs(loc = 5, scale = 2, size = n)

samples_sort = sorted(samples)

x_labels_p = np.arange(1/(2*n), 1, 1/n)
y_labels_p = stats.norm.cdf(samples_sort, loc = 5, scale = 2)

plt.scatter(x_labels_p, y_labels_p)
plt.title('PP plot for normal distribution samle')
plt.show()


x_labels_q = samples_sort
y_labels_q = stats.norm.ppf(x_labels_p, loc = 5, scale = 2)

plt.scatter(x_labels_q, y_labels_q)
plt.title('QQ plot for normal distribution samle')
plt.show()

stats.probplot(samples,plot=plt) 
plt.show() 



# Set the parameters
numData = 1000
myMean = 0
mySD = 3
 

   
# To get reproducable values, I provide a seed value
np.random.seed(1234)   
    
# Generate and show random data
data = stats.norm.rvs(myMean, mySD, size=numData)
fewData = data[:100]
plt.hist(data)
plt.show()

plt.hist(fewData)
plt.show()

 # --- >>> START stats <<< ---
# Graphical test: if the data lie on a line, they are pretty much normally distributed
_ = stats.probplot(data, plot=plt)
plt.show()

pVals = pd.Series()
pFewVals = pd.Series()
# The scipy normaltest is based on D-Agostino and Pearsons test that
# combines skew and kurtosis to produce an omnibus test of normality.
_, pVals['Omnibus']    = stats.normaltest(data)
_, pFewVals['Omnibus'] = stats.normaltest(fewData)

# Shapiro-Wilk test
_, pVals['Shapiro-Wilk']    = stats.shapiro(data)
_, pFewVals['Shapiro-Wilk'] = stats.shapiro(fewData)
    
    
# Alternatively with original Kolmogorov-Smirnov test
_, pVals['Kolmogorov-Smirnov']    = stats.kstest((data-np.mean(data))/np.std(data,ddof=1), 'norm')
_, pFewVals['Kolmogorov-Smirnov'] = stats.kstest((fewData-np.mean(fewData))/np.std(fewData,ddof=1), 'norm')
    
print('p-values for all {0} data points: ----------------'.format(len(data)))
print(pVals)
print('p-values for the first 100 data points: ----------------')
print(pFewVals)
    
if pVals['Omnibus'] > 0.05:
     print('Data are normally distributed')
# --- >>> STOP stats <<< ---
 
    
    
##hypothesis test
scores = np.array([109.4,76.2,128.7,93.7,85.6,117.7,117.2,87.3,100.3,55.1])

stats.normaltest(scores)

tval = (110-np.mean(scores))/stats.sem(scores)
td = stats.t(len(scores)-1)
p = 2*td.sf(tval)

""""
Chapter 8
"""
'''Data from Altman, check for significance of mean value.
'''
    
# Get data from Altman
inFile = 'altman_91.txt'
data = np.genfromtxt(inFile, delimiter=',')

# Watch out: by default the standard deviation in numpy is calculated with ddof=0, corresponding to 1/N!
myMean = np.mean(data)
mySD = np.std(data, ddof=1)     # sample standard deviation
print(('Mean and SD: {0:4.2f} and {1:4.2f}'.format(myMean, mySD)))

# Confidence intervals
tf = stats.t(len(data)-1)
# multiplication with np.array[-1,1] is a neat trick to implement "+/-"
ci = np.mean(data) + stats.sem(data)*np.array([-1,1])*tf.ppf(0.975)
print(('The confidence intervals are {0:4.2f} to {1:4.2f}.'.format(ci[0], ci[1])))

# Check if there is a significant difference relative to "checkValue"
checkValue = 7725
# --- >>> START stats <<< ---
t, prob = stats.ttest_1samp(data, checkValue)
if prob < 0.05:
     print(('{0:4.2f} is significantly different from the mean (p={1:5.3f}).'.format(checkValue, prob)))

# For not normally distributed data, use the Wilcoxon signed rank sum test
(rank, pVal) = (data-checkValue)
if pVal < 0.05:
   issignificant = 'unlikely'
else:
   issignificant = 'likely'
    # --- >>> STOP stats <<< ---
      
print(('It is ' + issignificant + ' that the value is {0:d}'.format(checkValue)))
    

##Two Group
np.random.seed(1234)
data = np.random.randn(10)+0.1
data1 = np.random.randn(10)*5
data2 = data1+data

stats.ttest_1samp(data,0)

stats.ttest_rel(data2,data1)


G = ((np.mean(data)-np.mean(data2))-0)/(np.sqrt(((1/500+1/500)((500-1)*np.std(data,ddof=1)+(500-1)*np.std(data2,ddof=1)))/(500+500-2)))





