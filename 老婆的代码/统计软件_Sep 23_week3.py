"""
统计软件与算法
2020年9月23日
"""

"""
循环语句
""" 
import numpy as np
#while loop
error = 50

while error >1:
    error = error/4
    print(error)
    
#for loop
    
#for loop over list
v = [1.73,1.68,1.71,1.89]
for var in v:
    print(var)
#loop over string   
for c in "family":
    print(c.capitalize(),end = '')

 # areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

i = 0
for a in areas:
    print(i,a)
    i+=1

# Change for loop to use enumerate() and update print()
for index, a in enumerate(areas) :
    print("room"+str(index)+":"+str(a))  
    
    
#for loop over list of list
house = [["hallway", 11.25], 
         ["kitchen", 18.0], 
         ["living room", 20.0], 
         ["bedroom", 10.75], 
         ["bathroom", 9.50]]
         
for i in house:
    print(i)
    print("the "+i[0]+" is "+str(i[1])+" sqm")

for i in house:
        for j in i:
            print(j, end= ' ')
        print("")   
    
#loop over dict
world  = {'a':30.55,'b':2.77,'c':39.21}
for key, value in world.items():
    print(key +":"+ str(value))
    
#loop over numpy array
height = [1.73,1.68,1.71,1.89,1.79]

np_height = np.array(height)


for val in np_height:
    print(val)
    
np_2d = np.array([[1.73,1.68,1.71,1.89,1.79],
                 [65.4,59.2,63.6,88.4,68.7]])
np_2d = np.array([[1.73,1.68],
                 ['a','b']])
    
for val in np_2d:
    print(val)

for val in np.nditer(np_2d):
    print(val)
    
for val in np.nditer(np_2d,order = 'C'):    #by default order = C
    print(val)   

for val in np.nditer(np_2d,order = 'F'):    
    print(val)
    

#break continue
var = 5                   
while var > 0:              
   var = var -1
   print('当前变量值 :', var)
print("Good bye!")
   
#1.break   
var = 5                   
while var > 0:              
   var = var -1
   if var == 2:             #when var =2, terminate the loop
      break
   print('当前变量值 :', var)
print("Good bye!")


#2.continue 
var = 5                   
while var > 0:              
   var = var -1
   if var == 2:
      continue
   print('当前变量值 :', var)
print("Good bye!")
 

"""
Pandas模块
"""

import pandas as pd


#Series
x = pd.Series(height)
print(x)

print(x.index)
print(x.values)

#change index
x = pd.Series(height, index=['a','b','c','d','e'])
print(x)

print(x['a'])   
x['a'] = 1.80   #modify values

#create Series using dict
world  = {'a':30.55,'b':2.77,'c':39.21}
y = pd.Series(world)
print(y)


#change index 
y1 = pd.Series(world,index=['a','b','d'])
print(y)    #check values for 'd'

#calculation
print(y[y>20])
print(y[:2])
print(y*2+np.exp(y))


#DataFrame
#build DF from dict
dict = {'country':['China','Russia','India'],
        'area':[9.597,17.10,3.286],
        'pop':[1357,143.5,1252]}

world2 = pd.DataFrame(dict)
print(world2)

#change lable
world2.index=['CH','RU','IN']
print(world2)

#build DF from array
w = pd.DataFrame(np.random.randn(7,3))
print(w)

w.columns = ['X1','X2','Y']
print(w)


print(w.size)
print(w.describe())
print(w.head())
print(w.tail())

#Index and select data 

#method1:square brackets
#col access
world2["country"]
type(world2['country'])

world2[["country"]]
type(world2[["country"]])

world2[['country','pop']]

#row access
world2[0:2]

world2[0:1]   #world2[0] will get error

#row and col access
world2[0:2][['country','pop']]


#method2:loc iloc
#row access loc
world2.loc['RU']        #Series
world2.loc[['RU']]          #DataFrame

world2.loc[['RU','CH']]

#row and col loc
world2.loc[['IN','CH'],['country','pop']]

world2.loc[:,['country','pop']]

#row access iloc
world2.iloc[[1]]

#row and col loc
world2.iloc[[2,0],[0,2]]
world2.iloc[:,[0,2]]

#Grouping
data = pd.DataFrame({'Gender':['f','f','m','f','m','m','f','m','f','m','m'],
                     'TV':[3.4,3.5,2.6,4.7,4.1,4.1,5.1,3.9,3.7,2.1,4.3]})

grouped = data.groupby('Gender')

print(grouped.describe())

df_female = grouped.get_group('f')  #get DF with gender = F
values_male=grouped.get_group('m').values #get array type data

print(df_female)
print(values_male)


#mathmatical calculation
np.random.seed(1010)
w = pd.DataFrame(np.random.randn(7,4))
v = pd.DataFrame(np.random.randn(5,3))
print(w)
print(v)

print(w.T)  #transpose
print(w+v)
print(v-v.iloc[0])
print(w.T@w)

print(w.sum(axis=0),'\n',w.sum(axis=1))


"""
#import data 
"""
import os
os.chdir('C:/Users/raonan/Desktop');     #更改路径
print(os.getcwd())          #查看当前路径

ls      #查看目录下文件   

#input data
x1 = input('Enter a name:')         #string
x2 = eval(input('Enter a number:'))         #int or float



#txt data
file = 'data1.txt'
p = open(file)
p.name
p.read(12)
p.mode


data = np.loadtxt(file,delimiter = ',')     #create array
data


df = pd.read_csv(file)
df = pd.read_csv(file,header = None)    #create DF
df.head()
df.iloc[[0],[0]]

df2 = pd.read_csv('data2.txt',usecols=['ID'])





