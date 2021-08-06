"""
统计软件与算法
2020年9月30日
"""

"""
#import data 
"""
import numpy as np
import pandas as pd


import os
os.chdir('C:/Users/raona/Desktop');     #更改路径
print(os.getcwd())          #查看当前路径

ls      #查看目录下文件   


#open file
file = open('data2.txt',mode = 'r') #open the file,mode is read 
print(file.read())  #print file 
file.close()    #close file

file.closed     #check whether it closed or not

#writing to a file
file1 = open('new.txt', mode = 'w')     #create a empty txt file, mode is writing 
file1.write('A message') #write 'A message' into file

file1.close()   #close file

##With As
with open('data2.txt') as O:
    print(O.read())
    print(O.closed)
    
print(O.closed)


#use NumPy open file
data1 = np.loadtxt('data2.txt',delimiter = ',')     #create array
data1 = np.loadtxt('data2.txt',delimiter = ',',skiprows = 1)     #create array
data1

data1[0]#first row
data1[0][2]

#use Pandas open file
df = pd.read_csv('data2.txt')
df = pd.read_csv('data2.txt',header = None)    #create DF
df.head()
df.iloc[[0],[0]]

df2 = pd.read_csv('data2.txt',index_col=0)
df2.head()

df_array = df.values        #convert DF to array
print(df_array)
type(df_array)


"""
Matplotlib
"""
import matplotlib.pyplot as plt

year = [1950, 1970, 1990, 2010]
pop = [2.519,3.692,5.263,6.972]

plt.plot(year,pop)
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('World Population')
plt.yticks([0,2,4,6,8,10])
plt.show()


#draw several lines in one figure
x = np.arange(0,2,0.05)
plt.figure(figsize = (10,5))
plt.plot(x,np.cos(x),'r',x,np.cos(x**2),'b^',x,np.cos(x**3),'g-.',x,np.cos(x**4),'mo',
         linewidth = 2.5,markersize = 5)
plt.ylim((-1.5,1.5))
plt.title('4 curves in one figure')


#show several graphes in one figure
fig,axs = plt.subplots(nrows = 2, ncols = 2)
axs[0,0].plot(x,np.cos(x),'r')
axs[0,1].plot(x,np.cos(x**2),'b^')
axs[1,0].plot(x,np.cos(x**3),'g-.')
axs[1,1].plot(x,np.cos(x**4),'mo')


##Other figure type
#scatter
plt.scatter(year,pop)
plt.show()

#pie 
y = np.random.rand(5)
y = y/sum(y)
y[y<0.05] = 0.05

plt.pie(y)
plt.title('Pie plot')


#histogram
x = np.random.randn(100)
plt.hist(x,bins = 30,label = 'Empirical')



"""
Function
"""
#use function to import several data
def load_file1 (X):
    df = pd.read_csv(X)
    print(df.head())
    #return df.head()


load_file1('data1.txt')
load_file1('data2.txt')


##位置参数
def f1 (x,y):
    print(x,y)
    
def f2(x,y):
    print(y,x)
    
f1(1,2)
f2(y= 2,x =1)

##默认参数
def f3(x,n = 10):
    print(x+n)

f3(2)
f3(2,1)

##可变参数
def f4(first, *second):
    print(first)
    print(second)
    
f4('a')
f4('a',1,2,3)


def f5(first, **dict):
    print(str(first) + "\n")
    print(dict)

f5(100, name = "tyson", age = "99") 


##* and **
def total(a=5, *numbers, **phonebook):
    print('a', a)

    #遍历元组中的所有项目
    for single_item in numbers:
        print('single_item', single_item)

    #遍历字典中的所有项目
    for first_part, second_part in phonebook.items():
        print(first_part,second_part)

print(total(10,1,2,3,Jack=1123,John=2231,Inge=1560))


##可变对象 不可变对象
#example1
test_list = [1, 2, 3, 4]
test_int = 3


def change(alist):
    alist[0] = 99
    print(alist)

def not_change(aint):
    aint = aint+90
    print(aint)

change(test_list)
not_change(test_int)
print(test_list)  # 改变了原来的值
print(test_int)  # 没有变

#example2
def f6 (var = []):
    var.append(1)
    print(var)

f6()
f6()
f6()

def f7(var = None):
    if var is None:
        var = []
    var.append(1)
    print(var)

f7()
f7()
f7([1,2])
f7([1,2])


##write function as a .py file
import function1   #file name
function1.func(2)   #function name



###Scipy
import scipy.stats as stats

#cdf example
x = np.arange(-4,4,0.01)
plt.plot(x,stats.norm.cdf(x))
plt.title('cdf of $N(0,1):\Phi(x)$')


stats.norm.rvs(size = 10,random_state = 1010, loc = 5,scale = 2)

#statistical test
a = np.random.normal(0,1,size = 100)
b = np.random.normal(1,1,size = 10)

stats.ttest_ind(a,b)











