import pandas as pd

data = pd.read_csv('重庆市各区县数字经济统计数据.csv')

# 相关性分析，计算各个指标之间可能存在的相关性
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rc("font",family='FangSong')
numeric_data = data.select_dtypes(include=['float64', 'int64'])

correlation = numeric_data.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5, annot_kws={"size": 15})
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.show()

# 重庆市各区县指标数据可视化

columns_of_interest = ['重庆市各区县/2023', '移动电话普及率', '电信业务总量/亿元', '数字惠普金融指数/2022', '研发投入/亿元/2022', '互联网普及率']
data_filtered = data[columns_of_interest]


data_filtered.set_index('重庆市各区县/2023', inplace=True)
data_filtered.index.name = None

# data_sorted = data_filtered.sort_values(by=['研发投入/亿元/2022'], ascending=False)

# 移动电话普及率
plt.figure(figsize=(12, 6))
data_filtered['移动电话普及率'].plot(kind='bar', color='#bc5559')
plt.axhline(y=1.19, color='#FEDC5E', linestyle='--', label='全国移动电话普及率')
plt.axhline(y=1.34, color='#FF6F48', linestyle='--', label='重庆市移动电话普及率')
plt.xticks(fontsize=15, rotation=45, ha='right')
plt.yticks(fontsize=15)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()

# 电信业务总量/亿元
plt.figure(figsize=(12, 6))
data_filtered['电信业务总量/亿元'].plot(kind='bar', color='#fae385')
plt.xticks(fontsize=15, rotation=45, ha='right')
plt.yticks(fontsize=15)
plt.legend(['电信业务总量/亿元'], fontsize=14)
plt.tight_layout()
plt.show()

# 数字惠普金融指数/2022
plt.figure(figsize=(12, 6))
data_filtered['数字惠普金融指数/2022'].plot(kind='bar', color='#fcc3c5')
plt.axhline(y=116.76, color='#FEDC5E', linestyle='--', label='全国数字惠普金融指数')
plt.xticks(fontsize=15, rotation=45, ha='right')
plt.yticks(fontsize=15)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()

# 研发投入/亿元/2022
plt.figure(figsize=(12, 6))
data_filtered['研发投入/亿元/2022'].plot(kind='bar', color='#d2dafe')
plt.legend(['研发投入/亿元/2022'], fontsize=14)
plt.xticks(fontsize=15, rotation=45, ha='right')
plt.yticks(fontsize=15)
plt.tight_layout()
plt.show()

# 互联网普及率
plt.figure(figsize=(12, 6))
data_filtered['互联网普及率'].plot(kind='bar', color='#5f6fbc')
plt.axhline(y=0.775, color='#FEDC5E', linestyle='--', label='全国互联网普及率')
plt.axhline(y=1.639, color='#FF6F48', linestyle='--', label='重庆市互联网普及率')
plt.xticks(fontsize=15, rotation=45, ha='right')
plt.yticks(fontsize=15)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()