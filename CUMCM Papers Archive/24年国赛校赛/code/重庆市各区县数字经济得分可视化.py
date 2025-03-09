import pandas as pd

data = pd.read_csv('重庆市各区县数字经济得分.csv')

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc("font",family='FangSong')

data.set_index('重庆市各区县/2023', inplace=True)
data.index.name = None

plt.figure(figsize=(12, 6))
data['得分'].sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.xticks(rotation=45, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(['得分'], fontsize=15)
plt.show()