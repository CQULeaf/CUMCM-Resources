# 相关性分析
import numpy as np
a = [0.051360, 0.056074, 0.059327, 0.062796, 0.070551]
b = [535.2861248, 483.2803043, 438.7188661, 385.4460061, 287.8690332]

correlation_coefficient = np.corrcoef(a, b)[0, 1]
print("相关系数:", correlation_coefficient)