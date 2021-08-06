# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# problem 1
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import random
from scipy import stats
import numpy as np
import math
import pandas as pd
import os


def jch(a):
    sum = 1
    for i in range(1, a+1):
        sum *= i
    return sum


result = 0
for i in range(1, 11):
    result += jch(i)
print(result)


# %%
# problem 2
data = pd.read_csv("mites.txt", sep='\s+')
data.loc[:, "dose"].value_counts()
g_data = data.groupby(data.loc[:, "dose"]).apply(
    lambda x: x["n.dead"]/x["n.mites"])
print(g_data)
for i in range(0, 4):
    print(g_data[i].mean())


# %%
# problem 3
P = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.1, 0.2, 0.3],
              [0.3, 0.4, 0.1, 0.2], [0.2, 0.3, 0.4, 0.1]])
for i in range(0, 4):
    print(P[i].sum())
for i in [2, 3, 5, 10]:
    print(np.power(P, i))
print("规律：对矩阵求幂得到各元素自身求幂组成的矩阵")

I = np.identity(4)
A = I - np.linalg.inv(P.T)
A[3] = [1, 1, 1, 1]
x = np.linalg.solve(A, [0, 0, 0, 1])
x


# %%
# problem 4 a


def directpoly(x, c):
    l = len(c)
    if type(x) == int:

        sum = 0
        for i in range(0, l):
            sum += c[i]*pow(x, i)
        return sum
    else:
        n = x.size
        m = np.ones(n)
        for i in range(0, n):
            sum = 0
            for j in range(0, l):
                sum += c[j]*pow(x[i], j)
            m[i] = sum
        return m


directpoly(np.array([1, 2, 3]), [1, 2, 3])


# %%
# problem 4 b


def honerpoly(x, c):
    lc = len(c)
    lx = x.size
    result = np.ones(lx)
    a = np.ones(lc)
    a[-1] = c[-1]
    for i in range(0, lx):
        for j in range(lc-2, -1, -1):
            a[j] = a[j+1]*x[i] + c[j]
        result[i] = a[0]
    return result


honerpoly(np.array([1, 2, 99]), np.array(
    [999, 2, 1, 1, 23, 123, 1, 231, 23, 123, 1, 231, 9]))


# %%
# problem 5
a = random.randint(0, 2)
b = random.randint(0, 2)
mu = [-1, 0, 1]
sigma = [0.5, 1, 1.5]
stats.norm(mu[a], sigma[b])
sampleNo = 100000
np.random.seed(0)
s = np.random.normal(mu[a], sigma[b], sampleNo)
plt.hist(s, bins=100, normed=True)
plt.show()


# %%
id = "statistics"
password = "2020"
num = 2
while num > 0:
    a = input("ID:")
    b = input("password:")
    if a == id and b == password:
        print("welcome!")
    else:
        num -= 1
        print("You can try only ", num, " times!")
        continue

# %%
