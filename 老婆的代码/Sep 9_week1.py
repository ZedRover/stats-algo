"""
统计软件与算法
2020年9月9日
"""

"""
PYTHON INTRODUCTION
"""
# Addition, subtraction
print(5 + 5)
print(5 - 5)

# Multiplication, division, modulo, and exponentiation
print(3 * 5)
print(10 / 2)
print(18 % 7)
print(4 ** 2)

# How much is your $100 worth after 7 years with 10% return each year?

print(100*1.1**7)
"""
Variables and Data Types
"""

#int,float,str,bool
height = 1.78
weight = 65

bmi = weight /height**2
print(bmi)

type(height)

x = 'height '
y = 'weight'
type(x)

z = True
type(z)

#string add/multiplication
x+y
x*2
#covert type using int(), float(),str(),bool()

# Definition of pi_string
pi_string = "3.1415926"

# Convert pi_string into float: pi_float
pi_float = float(pi_string)

# Definition of savings and result
savings = 100
result = 100 * 1.10 ** 7

# Fix the printout
str(savings)
str(result)
print("I started with $" + str(savings) + " and now have $" + str(result) + ".")


"halo行"
"""
LIST
"""
h1 = [1.70,1.65,1.82,1.80,1.60]

h2 = ['a',1.70,'b',1.65,'c',1.82,'d',1.80,'e',1.60]

#list of list
h3 = [['a',170],['b',1.65],['c',1.82],['d',1.80],['e',1.60]]

#subsetting list
h2[0]
h2[9]
h2[-1]

h2[2:4]
h2[2:]
h2[:2]

h3[-1][1]

#list manupulation

#chaning list elements
h2[1] = 1.72
print(h2)

h2[0:2] = ['g',1.75]
print(h2)

#adding and removing elements
h2_new = h2+['f',1.62]
print(h2_new)

#append extend del function
h2_new.append()
h2_new.extend(['h',1.88])
print(h2_new)

del(h2[2:4])
print(h2)

#copy of list
h1_copy = list(h1)
h1_copy = h1[:]

"""
DICTIONARY
"""
pop = [30.55,2.77,39.21]
countries = ['a','b','c']

world  = {'a':30.55,'b':2.77,'c':39.21}

world['a']

world['d'] = 2.78

'd' in world

del(world['d'])
print(world)
"""
FUNCTION and PACKAGES
"""
#build-in function
max(h1)
min(h1)

round(h1[1])
round(h1[1],1)

sorted(h1,reverse = True)

#list method
h1.index(1.82)          #call method index on h1
h1.count(1.82)

print(h1.index(1.82)/len(h1))

#string method
sister = 'liz'
sister.capitalize()
sister.upper()
sister.replace('z','sa')



"""
NUMPY
"""
import numpy as np

height = [1.73,1.68,1.71,1.89,1.79]
weight = [65.4,59.2,63.6,88.4,68.7]
weight/height**2 #get error

np_height = np.array(height)
np_weight = np.array(weight)

np_height
np_weight

bmi = np_weight/np_height**2
bmi

#remark: numpy array only contains one type

np_height+np_weight    #different type differnt behavior

#numpy array subsetting
bmi[1]

bmi>23

bmi[bmi>23]

#2D numpy array
np_2d = np.array([[1.73,1.68,1.71,1.89,1.79],
                 [65.4,59.2,63.6,88.4,68.7]])

np_2d

np_2d.shape

#2d numpy array subsetting
np_2d[0]

np_2d[0][2]
np_2d[0,2]

np_2d[:,1:3]

np_2d[1,:] 


#numpy basic statistics
np.mean(np_2d[0])
np.median(np_2d[1])

np.corrcoef(np_2d[0,:],np_2d[1,:])
np.std(np_2d[0,:])

#generate data
help(np.random.normal)
h_r = np.random.normal(1.75,0.2,5000)
w_r = np.random.normal(60,15,5000)