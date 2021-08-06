"""
统计软件与算法
2020年11月11日
"""

from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
from lifelines.datasets import load_waltons
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
os.chdir('/Users/zed/VSCode/Stats-Alg/')  # 更改路径

# pip install lifelines

# Generate some sample data, with a Weibull modulus of 1.5
WeibullDist = stats.weibull_min(1.5)
data = WeibullDist.rvs(500)

# Now fit the parameter
fitPars = stats.weibull_min.fit(data)

# Note: fitPars contains (WeibullModulus, Location, Scale)
print('The fitted Weibull modulus is {0:5.2f}, compared to the exact value of 1.5 .'.format(
    fitPars[0]))

''' Graphical representation of survival curves, and comparison of two
curves with logrank test.
"miR-137" is a short non-coding RNA molecule that functions to regulate
the expression levels of other genes.
'''

# Import standard packages


# Load and show the data
df = load_waltons()  # returns a Pandas DataFrame

print(df)
'''
    T  E    group
0   6  1  miR-137
1  13  1  miR-137
2  13  1  miR-137
3  13  1  miR-137
4  19  1  miR-137
'''

T = df['T']
E = df['E']

groups = df['group']
ix = (groups == 'miR-137')

kmf = KaplanMeierFitter()

kmf.fit(T[~ix], E[~ix], label='control')
ax = kmf.plot()

kmf.fit(T[ix], E[ix], label='miR-137')
kmf.plot(ax=ax)

plt.ylabel('Survival Probability')
outFile = 'lifelines_survival.png'


# Compare the two curves

results = logrank_test(
    T[ix], T[~ix], event_observed_A=E[ix], event_observed_B=E[~ix])
results.print_summary()


def correlation():
    '''Pearson correlation, and two types of rank correlation (Spearman, Kendall)
    comparing age and %fat (measured by dual-photon absorptiometry) for 18 normal adults.
    '''

    # Get the data
    inFile = 'altman_11_1.txt'
    data = np.genfromtxt(inFile, delimiter=',')
    x = data[:, 0]
    y = data[:, 1]

    # --- >>> START stats <<< ---
    # Calculate correlations
    # Resulting correlation values are stored in a dictionary, so that it is
    # obvious which value belongs to which correlation coefficient.
    corr = {}
    corr['pearson'], _ = stats.pearsonr(x, y)
    corr['spearman'], _ = stats.spearmanr(x, y)
    corr['kendall'], _ = stats.kendalltau(x, y)
    # --- >>> STOP stats <<< ---

    print(corr)

    # Assert that Spearman's rho is just the correlation of the ranksorted data
    np.testing.assert_almost_equal(corr['spearman'], stats.pearsonr(
        stats.rankdata(x), stats.rankdata(y))[0])

    return corr['pearson']  # should be 0.79208623217849117


correlation()
