"""
统计软件与算法
2020年9月16日
"""

"""
Remark(array subsetting)
"""
import numpy as np
#2D numpy array
np_2d = np.array([[1.73,1.68,1.71,1.89,1.79],
                 [65.4,59.2,63.6,88.4,68.7]])


#2d numpy array subsetting
np_2d[0]        #first row
np_2d[0,:]       #first row


np_2d[0:1,1:3]


np_2d[0][2]
np_2d[0,2]


np_2d[[0,1,1],  [0,3,1]]

#创建数组
x = np.empty((3,2), dtype =  int)  

y = np.zeros(5)

z = np.ones((2,4))

x = np.arange(5) 
x = np.arange(10,20,2)

x = np.linspace(10,20,5)



"""
数组变换
"""

# 创建一个行向量
vector = np.array([1, 2, 3, 4, 5, 6])

 
# 创建一个二维数组表示一个矩阵
matrix3 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
 
# 查看行数和列数
print(matrix3.shape)
 
# 查看元素数量
print(matrix3.size)
 
# 查看维数
print(matrix3.ndim)


# 调整大小  
matrix3.shape =  (4,3)  #直接改变了matrix3
print(matrix3)

a = matrix3.reshape(4,3)

# reshape如果提供一个整数，那么reshape会返回一个长度为该整数值的一维数组
print(matrix3.reshape(12))

c = matrix3.reshape(2,2,3)  
print(c)

#矩阵转置
vector = np.array([1, 2, 3, 4, 5, 6])
 
matrix_vector = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 转置matrix_vector矩阵
print(matrix_vector.T)
 
# 向量是不能被转置的
print(vector.T)
vector.T.shape

# 转置向量通常指二维数组表示形式下将行向量转换为列向量或者反向转换
print(np.array([[1, 2, 3, 4, 5, 6]]).T)

# 矩阵展开
print(matrix_vector.flatten())

#数组链接
a = np.array([[1,2],[3,4]]) 
 
b = np.array([[5,6],[7,8]]) 

print(np.concatenate((a,b))) 
print(np.concatenate((a,b),axis = 1))

print(np.hstack((a,b)))
print(np.vstack((a,b)))


a = np.array([[1,2,3],[4,5,6]]) 

#向a里加元素
print(np.append(a, [7,8,9])) 
 
print(np.append(a, [[7,8,9]],axis = 0))

print(np.append(a, [[5,5,5],[7,8,9]],axis = 1))

#删除第二列
print(np.delete(a,1,axis = 0))


"""
数组计算
"""

# 返回最大的元素
print(np.max(matrix3))
 
# 返回最小元素
print(np.min(matrix3))
 
# 找到每一列的最大元素
print(np.max(matrix3, axis=0))
 
# 找到每一行最大的元素
print(np.max(matrix3, axis=1))
 
# 返回平均值
print(np.mean(matrix3))
 
# 返回方差
print(np.var(matrix3))
 
# 返回标准差
print(np.std(matrix3))

# 求每一列的平均值
print(np.mean(matrix3, axis=0))
 
# 求每一行的方差
print(np.var(matrix3, axis=1))

# 点积
vector_a = np.array([1, 2, 3])
vector_b = np.array([4, 5, 6])

print(np.dot(vector_a, vector_b))

# Python 3.5+ 版本可以这样求
print(vector_a @ vector_b)
 
 
#矩阵加,减，乘，逆
matrix_c = np.array([[1, 1], [1, 2]])
matrix_d = np.array([[1, 3], [1, 2]])
 
print(np.add(matrix_c, matrix_d))
 
print(np.subtract(matrix_c, matrix_d))
 
print(matrix_c + matrix_d)
print(matrix_c - matrix_d)
  
# 两矩阵相乘
print(np.dot(matrix_c, matrix_d))
 
print(matrix_c @ matrix_d)
 
#矩阵对应元素相乘，而非矩阵乘法
print(matrix_c * matrix_d)
 

matrix_e = np.array([[1, 4], [2, 5]])
 
# 计算矩阵的逆
print(np.linalg.inv(matrix_e))
 
# 验证一个矩阵和它的逆矩阵相乘等于I（单位矩阵）
print(matrix_e @ np.linalg.inv(matrix_e))



"""
逻辑运算符
"""
#数值比较
2<3

2==3

3<=3

#布尔运算 and or not
x = 12
x>5 and x<15

y=5
y<7 or y>13


#numpy逻辑运算
a = np.array([21.5,15.8,25,18.2,17])
a>20
a>18 and a<22 #error

np.logical_and(a>18,a<22)
a[np.logical_and(a>18,a<22)]

#条件语句 if else elif
z = 4
if z%2 ==0: #True
    print('z is even')


z = 5
if z%2 ==0: #false
    print('z is even')
else:
    print('z is odd')
    
    
z = 3
if z%2 ==0: #false
    print('z is even')
elif z%3 ==0:
    print('z is divisible by 3')
else:
    print('z is neither divisible by 2 nor by 3')


"""
循环语句
""" 
#while loop
error = 50

while error >1:
    error = error/4
    print(error)
    
