import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

mpl.rcParams['font.family'] = 'Palatino Linotype'

def calculate_n(p):
    return (0.4935 / (p - 0.1)) ** 2

# 给定 p 值求对应的 n 值
p = 0.15
n = calculate_n(p)
print(f'p = {p}, n = {n}')

p_values = np.linspace(0.11, 1, 400)
n_values = calculate_n(p_values)

# 主图设置
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(p_values, n_values, label=r'${n_1} \geq \left( \frac{0.4935}{\hat{p_1} - 0.1} \right)^2$', color='#708194')

ax.set_xlabel(r'$\hat{p_1}$', fontsize=20)
ax.set_ylabel('$n_1$', fontsize=20)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.legend(fontsize=18)
ax.grid(True)

# 创建放大区域
axins = inset_axes(ax, width="50%", height="50%", loc='lower right',
                   bbox_to_anchor=(0.2, 0.35, 0.7, 0.7),
                   bbox_transform=ax.transAxes)

# 在放大区域中绘制相同的函数
x_zoomed = np.linspace(0.11, 0.20, 400)
y_zoomed = calculate_n(x_zoomed)
axins.plot(x_zoomed, y_zoomed, color='#8ea0aa')
axins.grid(True)

# 设置放大区域的x轴和y轴限制
axins.set_xlim(0.14, 0.16)
axins.set_ylim(70, 150)
axins.tick_params(axis='x', labelsize=14)
axins.tick_params(axis='y', labelsize=14)

# 添加放大区域和主图之间的连接线
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.show()

def calculate_n(p):
    return (0.3846 / (0.1 - p)) ** 2

p_values = np.linspace(0, 0.09, 400)
n_values = calculate_n(p_values)

# 给定p值求对应的n值
p = 0.06
n = calculate_n(p)
print(f'p = {p}, n = {n}')

# 主图设置
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(p_values, n_values, label=r'${n_2} \geq \left( \frac{0.3846}{0.1 - \hat{p_2}} \right)^2$', color='#708194')

ax.set_xlabel(r'$\hat{p_2}$', fontsize=20)
ax.set_ylabel('$n_2$', fontsize=20)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.legend(fontsize=18)
ax.grid(True)

# 创建放大区域，位置和大小通过参数调整
axins = inset_axes(ax, width="50%", height="50%", loc='lower right',
                   bbox_to_anchor=(-0.15, 0.3, 0.7, 0.7),
                   bbox_transform=ax.transAxes)

# 在放大区域中绘制相同的函数
x_zoomed = np.linspace(0, 0.09, 400)
y_zoomed = calculate_n(x_zoomed)
axins.plot(x_zoomed, y_zoomed, color='#8ea0aa')
axins.grid(True)

# 设置放大区域的x轴和y轴限制
axins.set_xlim(0.05, 0.07)
axins.set_ylim(50, 175)
axins.tick_params(axis='x', labelsize=14)
axins.tick_params(axis='y', labelsize=14)

# 添加放大区域和主图之间的连接线
mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")

plt.show()