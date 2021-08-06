"""
统计软件与算法
2020年10月28
"""

from scipy.stats.mstats import kruskalwallis
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
os.chdir('C:/Users/raona/Desktop')  # 更改路径

"""
CHAPTER 8
"""
# One sample t-test
inFile = 'altman_91.txt'
data = np.genfromtxt(inFile, delimiter=',')

myMean = np.mean(data)
mySD = np.std(data, ddof=1)     # sample standard deviation

checkValue = 7725

stats.ttest_1samp
t, prob = stats.ttest_1samp(data, checkValue)
print(prob)


# One sample Wilcoxon test
x = np.array([1, 4, 2, 8, 5, 7])
print(np.sort(x))
print(stats.rankdata(x))

# Two Group dependent
np.random.seed(1234)
data = np.random.randn(10)+0.1
data1 = np.random.randn(10)*5
data2 = data1+data

stats.ttest_1samp(data, 0)

stats.ttest_rel(data2, data1)

# independent


# Mann-Whitney
stats.mannwhitneyu
stats.ranksums

np.random.seed(1010)
x = np.random.normal(3, 1, 500)
y = np.random.normal(3.2, 1, 500)
print(stats.ranksums(x, y))
print(stats.mannwhitneyu(x, y, alternative="two-sided"))

# EXAMPLE(two groups)


def paired_data():
    '''Analysis of paired data
    Compare mean daily intake over 10 days before taking the medicine and 10 days after taking the medicine (in kJ).'''

    # Get the data:  daily intake of energy in kJ for 11 person
    inFile = 'altman_93.txt'
    data = np.genfromtxt(inFile, delimiter=',')

    np.mean(data, axis=0)
    np.std(data, axis=0, ddof=1)

    pre = data[:, 0]
    post = data[:, 1]

    # --- >>> START stats <<< ---
    # paired t-test: doing two measurments on the same experimental unit
    # e.g., before and after a treatment
    t_statistic, p_value = stats.ttest_1samp(post - pre, 0)

    # p < 0.05 => alternative hypothesis:the difference in mean is not equal to 0
    print(("paired t-test", p_value))

    # alternative to paired t-test when data has an ordinary scale or when not normally distributed
    rankSum, p_value = stats.wilcoxon(post - pre)
    # --- >>> STOP stats <<< ---
    print(("Wilcoxon-Signed-Rank-Sum test", p_value))

    return p_value


paired_data()


def unpaired_data():
    ''' Then some unpaired comparison: 24 hour total energy expenditure (MJ/day),
    in groups of lean and obese person'''

    # Get the data: energy expenditure in mJ and stature (0=obese, 1=lean)
    inFile = 'altman_94.txt'
    energ = np.genfromtxt(inFile, delimiter=',')

    # Group them
    group1 = energ[:, 1] == 0
    group1 = energ[group1][:, 0]
    group2 = energ[:, 1] == 1
    group2 = energ[group2][:, 0]

    np.mean(group1)
    np.mean(group2)
    group1 = np.random.normal(3, 1, 5000)
    group2 = np.random.normal(3.2, 1, 5000)
    # --- >>> START stats <<< ---
    # two-sample t-test
    # null hypothesis: the two groups have the same mean
    t_statistic, p_value = stats.ttest_ind(group1, group2)

    # p_value < 0.05 => alternative hypothesis:
    print(("two-sample t-test", p_value))

    # For non-normally distributed data, perform the two-sample wilcoxon test
    # a.k.a Mann Whitney U
    u, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')

    print(("Mann-Whitney test", p_value))
    # --- >>> STOP stats <<< ---

    # Plot the data
    plt.plot(group1, 'bx', label='obese')
    plt.plot(group2, 'ro', label='lean')
    plt.legend(loc=0)
    plt.show()


unpaired_data()

''' Analysis of Variance (ANOVA)
- Levene test
- ANOVA - oneway
- Do a simple one-way ANOVA, using statsmodels
- Show how the ANOVA can be done by hand.
- For the comparison of two groups, a one-way ANOVA is equivalent to
  a T-test: t^2 = F
'''

# additional packages


def anova_oneway():
    ''' One-way ANOVA: test if results from 3 groups are equal.

    Twenty-two patients undergoing cardiac bypass surgery were randomized to one of three ventilation groups:

    Group I: Patients received 50% nitrous oxide and 50% oxygen mixture continuously for 24 h.
    Group II: Patients received a 50% nitrous oxide and 50% oxygen mixture only dirng the operation.
    Group III: Patients received no nitrous oxide but received 35-50% oxygen for 24 h.

    The data show red cell folate levels for the three groups after 24h' ventilation.

    '''
    # Get the data
    print('One-way ANOVA: -----------------')
    inFile = 'altman_910.txt'
    data = np.genfromtxt(inFile, delimiter=',')

    # Sort them into groups, according to column 1
    group1 = data[data[:, 1] == 1, 0]
    group2 = data[data[:, 1] == 2, 0]
    group3 = data[data[:, 1] == 3, 0]

    # --- >>> START stats <<< ---
    # First, check if the variances are equal, with the "Levene"-test
    (W, p) = stats.levene(group1, group2, group3)
    if p < 0.05:
        print(
            ('Warning: the p-value of the Levene test is <0.05: p={0}'.format(p)))

    # Do the one-way ANOVA
    F_statistic, pVal = stats.f_oneway(group1, group2, group3)
    # --- >>> STOP stats <<< ---

    # Print the results
    print('Data form Altman 910:')
    print((F_statistic, pVal))
    if pVal < 0.05:
        print('One of the groups is significantly different.')

    # Elegant alternative implementation, with pandas & statsmodels
    df = pd.DataFrame(data, columns=['value', 'treatment'])
    model = ols('value ~ C(treatment)', df).fit()
    anovaResults = anova_lm(model)
    print(anovaResults)

    # should be (3.711335988266943, 0.043589334959179327)
    return (F_statistic, pVal)


# ----------------------------------------------------------------------
def show_teqf():
    """Shows the equivalence of t-test and f-test, for comparing two groups"""

    # Get the data
    data = pd.read_csv('galton.csv')

    # First, calculate the F- and the T-values, ...
    F_statistic, pVal = stats.f_oneway(data['father'], data['mother'])
    t_val, pVal_t = stats.ttest_ind(data['father'], data['mother'])

    # ... and show that t**2 = F
    print('\nT^2 == F: ------------------------------------------')
    print(
        ('From the t-test we get t^2={0:5.3f}, and from the F-test F={1:5.3f}'.format(t_val**2, F_statistic)))

    # numeric test
    np.testing.assert_almost_equal(t_val**2, F_statistic)

    return F_statistic


'''Multiple testing'''

inFile = 'altman_910.txt'
data = np.genfromtxt(inFile, delimiter=',')
# Sort them into groups, according to column 1
group1 = data[data[:, 1] == 1, 0]
group2 = data[data[:, 1] == 2, 0]
group3 = data[data[:, 1] == 3, 0]

# Then, do the multiple testing
multiComp = MultiComparison(data[:, 0], data[:, 1])
print((multiComp.tukeyhsd().summary()))


# Calculate the p-values:
res2 = pairwise_tukeyhsd(data[:, 0], data[:, 1])
df = pd.DataFrame(data)
numData = len(df)
numTreatments = 3
dof = numData - numTreatments

# Show the group names
print((multiComp.groupsunique))

# Get the data
xvals = np.arange(3)
res2 = pairwise_tukeyhsd(data[:, 0], data[:, 1])
errors = np.ravel(np.diff(res2.confint)/2)

# Plot them
plt.plot(xvals, res2.meandiffs, 'o')
plt.errorbar(xvals, res2.meandiffs, yerr=errors, fmt='o')

# Put on labels
pair_labels = multiComp.groupsunique[np.column_stack(
    res2._multicomp.pairindices)]
plt.xticks(xvals, pair_labels)


def Holm_Bonferroni(multiComp):
    ''' Instead of the Tukey's test, we can do pairwise t-test '''

    # First, with the "Holm" correction
    rtp = multiComp.allpairtest(stats.ttest_rel, method='Holm')
    print((rtp[0]))

    # and then with the Bonferroni correction
    print((multiComp.allpairtest(stats.ttest_rel, method='b')[0]))

    # Any value, for testing the program for correct execution
    checkVal = rtp[1][0][0, 0]
    return checkVal


'''Example of a Kruskal-Wallis test (for not normally distributed data)
Taken from http://www.brightstat.com/index.php?option=com_content&task=view&id=41&Itemid=1&limit=1&limitstart=2
'''

# additional packages


def main():
    '''These data could be a comparison of the smog levels in four different cities. '''

    # Get the data
    city1 = np.array([68, 93, 123, 83, 108, 122])
    city2 = np.array([119, 116, 101, 103, 113, 84])
    city3 = np.array([70, 68, 54, 73, 81, 68])
    city4 = np.array([61, 54, 59, 67, 59, 70])

    # --- >>> START stats <<< ---
    # Perform the Kruskal-Wallis test
    h, p = kruskalwallis(city1, city2, city3, city4)
    # --- >>> STOP stats <<< ---

    # Print the results
    if p < 0.05:
        print('There is a significant difference between the cities.')
    else:
        print('No significant difference between the cities.')

    return h


''' Two-way Analysis of Variance (ANOVA)
The model is formulated using the "patsy" formula description. This is very
similar to the way models are expressed in R.
'''

# additional packages


def anova_interaction():
    '''ANOVA with interaction: Measurement of fetal head circumference,
    by four observers in three fetuses, from a study investigating the
    reproducibility of ultrasonic fetal head circumference data.
    '''

    # Get the data
    inFile = 'altman_12_6.txt'
    data = np.genfromtxt(inFile, delimiter=',')

    # Bring them in DataFrame-format
    df = pd.DataFrame(data, columns=['hs', 'fetus', 'observer'])

    # --- >>> START stats <<< ---
    # Determine the ANOVA with interaction
    formula = 'hs ~ C(fetus) + C(observer) + C(fetus):C(observer)'
    lm = ols(formula, df).fit()
    anovaResults = anova_lm(lm)
    # --- >>> STOP stats <<< ---
    print(anovaResults)
