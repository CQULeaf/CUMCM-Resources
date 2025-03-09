import pandas as pd
import numpy as np
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
# # 熵权法函数
# def entropy_weight_method(matrix):
#     m, n = matrix.shape
#     print(m,n)
#     normalized_matrix = matrix / np.sum(matrix, axis=0)
#     log_matrix = -np.log(normalized_matrix + 1e-10)
#     entropy = np.sum(normalized_matrix * log_matrix, axis=1)
#     entropy_weight = (1 - entropy) / np.sum(1 - entropy)
#     return entropy_weight
import numpy as np  # 导入numpy库，并简称为np
 
# 定义一个自定义的对数函数mylog，用于处理输入数组中的零元素
def mylog(p):
    n = len(p)  # 获取输入向量p的长度
    lnp = np.zeros(n)  # 创建一个长度为n，元素都为0的新数组lnp
    for i in range(n):  # 对向量p的每一个元素进行循环
        if p[i] == 0:  # 如果当前元素的值为0
            lnp[i] = 0  # 则在lnp中对应位置也设置为0，因为log(0)是未定义的，这里我们规定为0
        else:
            lnp[i] = np.log(p[i])  # 如果p[i]不为0，则计算其自然对数并赋值给lnp的对应位置
    return lnp  # 返回计算后的对数数组
csv_file='重庆市数据最终版.csv'
data=pd.read_csv(csv_file)
#print(data)
indicators=data[['移动电话普及率','电信业务总量/亿元','数字惠普金融指数/2022','研发投入/亿元/2022','互联网普及率']]
#print(indicators)
#相关性分析
X = indicators.values
result2 = np.corrcoef(X, rowvar=False)
# print(result2)
# 相关性分析分析结果较强，可以考虑使用主成分分析

 
# kmo_all, kmo_model = calculate_kmo(X)
# print(kmo_all)
#print(X)
# # 计算权重
# weights = entropy_weight_method(matrix)
# print("权重:", weights)
Z = X / np.sqrt(np.sum(X*X, axis=0))
#print("标准化矩阵 Z = ")
#print(Z)  # 打印标准化矩阵Z
# 计算熵权所需的变量和矩阵初始化
n, m = Z.shape  # 获取标准化矩阵Z的行数和列数
D = np.zeros(m)  # 初始化一个长度为m的数组D，用于保存每个指标的信息效用值
# 计算每个指标的信息效用值
for i in range(m):  # 遍历Z的每一列
    x = Z[:, i]  # 获取Z的第i列，即第i个指标的所有数据
    p = x / np.sum(x)  # 对第i个指标的数据进行归一化处理，得到概率分布p
    # 使用自定义的mylog函数计算p的对数。需要注意的是，如果p中含有0，直接使用np.log会得到-inf，这里使用自定义函数避免这个问题
    e = -np.sum(p * mylog(p)) / np.log(n)  # 根据熵的定义计算第i个指标的信息熵e
    D[i] = 1 - e  # 根据信息效用值的定义计算D[i]
# 根据信息效用值计算各指标的权重
W = D / np.sum(D)  # 将信息效用值D归一化，得到各指标的权重W
#print("权重 W = ")
print(W)  # 打印得到的权重数组W
weighted_average = Z @ W
# len_w = len(weighted_average)
# sum_w = sum(weighted_average)
# for i in range(len_w):
#     weighted_average[i] /= sum_w
print("加权平均值为:")
print(weighted_average)