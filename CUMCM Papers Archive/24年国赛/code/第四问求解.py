import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

mpl.rcParams['font.family'] = 'Palatino Linotype'

# 定义函数 hat_p1(变量 n1)
def calculate_hat_p1(n1):
    return 0.1 + 0.4935 / np.sqrt(n1)

# 当n1取某一个值时，求出对应的 hat_p1 的值
n1 = 100
hat_p1 = calculate_hat_p1(n1)
print(f'n = {n1}, p = {hat_p1}')

# 生成 n1 的取值范围
n1_values = np.linspace(1, 1000, 400)  # 避免 sqrt(0)，所以 n1 从 1 开始
hat_p1_values = calculate_hat_p1(n1_values)

# 主图设置
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(n1_values, hat_p1_values, label=r'$\hat{p_1} \geq 0.1 + \frac{0.4935}{\sqrt{n_1}}$', color='#3c2827')

ax.set_xlabel(r'$n_1$', fontsize=20)
ax.set_ylabel(r'$\hat{p_1}$', fontsize=20)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.legend(fontsize=18)
ax.grid(True)

# 创建放大区域，位置和大小通过参数调整
axins = inset_axes(ax, width="50%", height="50%", loc='lower right',
                   bbox_to_anchor=(0.2, 0.4, 0.6, 0.6),
                   bbox_transform=ax.transAxes)

# 在放大区域中绘制相同的函数
n1_zoomed = np.linspace(90, 110, 400)  # 选择需要放大的区域
hat_p1_zoomed = calculate_hat_p1(n1_zoomed)
axins.plot(n1_zoomed, hat_p1_zoomed, color='#3c2827')
axins.grid(True)

# 设置放大区域的x轴和y轴限制
axins.set_xlim(90, 110)
axins.set_ylim(0.147, 0.1525)
axins.tick_params(axis='x', labelsize=12)
axins.tick_params(axis='y', labelsize=12)

# 添加放大区域和主图之间的连接线
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.show()

# 定义函数 hat_p2(变量 n2)
def calculate_hat_p2(n2):
    return 0.1 - 0.3846 / np.sqrt(n2)

# 生成 n2 的取值范围
n2_values = np.linspace(30, 1000, 400)  # 避免 sqrt(0)，所以 n2 从 1 开始
hat_p2_values = calculate_hat_p2(n2_values)

n2 = 100
hat_p2 = calculate_hat_p2(n2)
print(f'n = {n2}, p = {hat_p2}')

# 主图设置
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(n2_values, hat_p2_values, label=r'$\hat{p_2} \leq 0.1 - \frac{0.3846}{\sqrt{n_2}}$', color='#3c2827')

ax.set_xlabel(r'$n_2$', fontsize=20)
ax.set_ylabel(r'$\hat{p_2}$', fontsize=20)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.legend(fontsize=16)
ax.grid(True)

# 创建放大区域，位置和大小通过参数调整
axins = inset_axes(ax, width="50%", height="50%", loc='lower right',
                   bbox_to_anchor=(0.2, 0.15, 0.6, 0.6),
                   bbox_transform=ax.transAxes)

# 在放大区域中绘制相同的函数
n2_zoomed = np.linspace(90, 110, 400)  # 选择需要放大的区域
hat_p2_zoomed = calculate_hat_p2(n2_zoomed)
axins.plot(n2_zoomed, hat_p2_zoomed, color='#3c2827')
axins.grid(True)

# 设置放大区域的x轴和y轴限制
axins.set_xlim(90, 110)  # 放大你感兴趣的 n2 范围
axins.set_ylim(0.0595, 0.0635)  # 根据实际需要调整 y 轴范围
axins.tick_params(axis='x', labelsize=12)
axins.tick_params(axis='y', labelsize=12)

# 添加放大区域和主图之间的连接线
mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")

plt.show()

import random
import math

def profit(n1, n2, q1, q2, q3, a1, a2, a3, a4, a5, a6, a7, a8, a9, D1, D2, E, F):
    """
    计算利润总和、成品数量及更新后的次品率
    """
    # 零件检测成本
    c21 = n1 * D1 * a2 + n2 * D2 * a4

    # 更新零件数量，考虑是否检测
    n1 = (1 - D1 * q1) * n1
    n2 = (1 - D2 * q2) * n2

    if n1 >= n2:  # 如果零件1数量多于零件2
        c22 = n2 * E * a6  # 成品检测成本
        c2 = c22 + c21     # 总检测成本
        c3 = n2 * a5       # 装配成本
        c4 = F * (n2 - (1 - q3) * (1 - q2) * n2 * (1 - q1) / (1 - D1 * q1)) * a9  # 拆解费用
        c5 = (1 - E) * (n2 - (1 - q3) * (1 - q2) * n2 * (1 - q1) / (1 - D1 * q1)) * a8  # 调换成本
        w = (1 - q3) * (1 - q2) * n2 * (1 - q1) / (1 - D1 * q1) * a7  # 销售额
        A = c2 + c3 + c4 + c5   # 成本总和
        pi = w - A              # 利润总和

        # 更新次品率
        N = n2 - (1 - q3) * (1 - q2) * n2 * (1 - q1) / (1 - D1 * q1)
        q1_new = ((1 - q2) * n2 * (1 - D1) * q1 / (1 - D1 * q1) +
                  (1 - D2) * q2 * n2 * (1 - D1) * q1 * (1 - D1 * q1)) / N
        q2_new = ((1 - D2) * q2 * n2 * (1 - q1) / (1 - D1 * q1) +
                  (1 - D2) * q2 * n2 * (1 - D1) * q1 / (1 - D1 * q1)) / N
    else:  # 如果零件1数量少于零件2
        c22 = n1 * E * a6  # 成品检测成本
        c2 = c22 + c21     # 总检测成本
        c3 = n1 * a5       # 装配成本
        c4 = F * (n1 - (1 - q3) * (1 - q1) * n1 * (1 - q2) / (1 - D2 * q2)) * a9  # 拆解费用
        c5 = (1 - E) * (n1 - (1 - q3) * (1 - q1) * n1 * (1 - q2) / (1 - D2 * q2)) * a8  # 调换成本
        w = (1 - q3) * (1 - q1) * n1 * (1 - q2) / (1 - D2 * q2) * a7  # 销售额
        A = c2 + c3 + c4 + c5   # 成本总和
        pi = w - A              # 利润总和

        # 更新次品率
        N = n1 - (1 - q3) * (1 - q1) * n1 * (1 - q2) / (1 - D2 * q2)
        q2_new = ((1 - q1) * n1 * (1 - D2) * q2 / (1 - D2 * q2) +
                  (1 - D1) * q1 * n1 * (1 - D2) * q2 * (1 - D2 * q2)) / N
        q1_new = ((1 - D1) * q1 * n1 * (1 - q2) / (1 - D2 * q2) +
                  (1 - D1) * q1 * n1 * (1 - D2) * q2 / (1 - D2 * q2)) / N

    return pi, N, q1_new, q2_new

#q1,q2,q3,a1,a2,a3,a4,a5,a6,a7,a8,a9=0.15,0.15,0.15,4,2,18,3,6,3,56,6,5
#q1,q2,q3,a1,a2,a3,a4,a5,a6,a7,a8,a9=0.15,0.15,0.15,4,2,18,3,6,3,56,6,5
#q1,q2,q3,a1,a2,a3,a4,a5,a6,a7,a8,a9=0.15,0.15,0.15,4,2,18,3,6,3,56,30,5
#q1,q2,q3,a1,a2,a3,a4,a5,a6,a7,a8,a9=0.15,0.15,0.15,4,1,18,1,6,2,56,30,5
#q1,q2,q3,a1,a2,a3,a4,a5,a6,a7,a8,a9=0.15,0.15,0.15,4,8,18,1,6,2,56,10,5
#q1,q2,q3,a1,a2,a3,a4,a5,a6,a7,a8,a9=0.15,0.15,0.15,4,2,18,3,6,3,56,10,40

#q1,q2,q3,a1,a2,a3,a4,a5,a6,a7,a8,a9=0.06,0.06,0.06,4,2,18,3,6,3,56,6,5
#q1,q2,q3,a1,a2,a3,a4,a5,a6,a7,a8,a9=0.06,0.06,0.06,4,2,18,3,6,3,56,6,5
#q1,q2,q3,a1,a2,a3,a4,a5,a6,a7,a8,a9=0.06,0.06,0.06,4,2,18,3,6,3,56,30,5
#q1,q2,q3,a1,a2,a3,a4,a5,a6,a7,a8,a9=0.06,0.06,0.06,4,1,18,1,6,2,56,30,5
#q1,q2,q3,a1,a2,a3,a4,a5,a6,a7,a8,a9=0.06,0.06,0.06,4,8,18,1,6,2,56,10,5
q1,q2,q3,a1,a2,a3,a4,a5,a6,a7,a8,a9=0.06,0.06,0.06,4,2,18,3,6,3,56,10,40

# 主程序
max_pi = 0
lst_max = []
lst_pi = []  # 存储1000个利润
lst_decision = []  # 存储决策过程

for _ in range(1000):
    n1 = n2 = 1  # 初始零件数量
    pi = 0
    decisions = []  # 记录每次的决策状态

    # 随机选择决策
    D1 = random.choice([0, 1])
    D2 = random.choice([0, 1])
    E = random.choice([0, 1])
    F = random.choice([0, 1])
    decisions.extend([D1, D2, E, F])

    # 计算利润、数量和次品率
    pi1, N, q1_new, q2_new = profit(n1, n2, q1, q2, q3, a1, a2, a3, a4, a5, a6, a7, a8, a9, D1, D2, E, F)

    n = 0
    while F == 1:
        n1 = n2 = N  # 更新数量
        D1 = random.choice([0, 1])
        D2 = random.choice([0, 1])
        E = random.choice([0, 1])
        F = random.choice([0, 1])

        # 计算新一轮利润
        pi2, N, q1_new, q2_new = profit(n1, n2, q1_new, q2_new, q3, a1, a2, a3, a4, a5, a6, a7, a8, a9, D1, D2, E, F)

        if pi2 > 0:
            decisions.extend([D1, D2, E, F])
            pi1 += pi2
        else:
            break

        if n > 3:
            break
        n += 1

    # 记录每次模拟的结果
    lst_pi.append(pi1)
    lst_decision.append(decisions)

    # 更新最大利润
    if max_pi < pi1:
        max_pi = pi1
        lst_max = decisions

# 检查是否有更好的决策方案
for i in range(1000):
    if abs(max_pi - lst_pi[i]) / max_pi < 0.0025 and len(lst_max) > len(lst_decision[i]):
        max_pi = lst_pi[i]
        lst_max = lst_decision[i]

# 打印结果
print(max_pi)
print(lst_max)

import random
number=0
lst11=[]#存储决策
lst12=[]#存储利润
lst141,lst151=[],[]
for i in range(1000):
    lst131=[]
    
    # try:
    def three_1(n1,q1,G1,H1,I1,q4=0.06):
            #print(n1)
            c21=G1*(n1+n1+2*n1) #零件检测成本
            c31=(1-G1*q1)*8*n1 #半成品装配成本
            c22_1=H1*(1-G1*q1)*4*n1 #半成品检测成本
            c41_1=H1*I1*(1-G1*q1-((1-q1)**3)*(1-q4)/(1-G1*q1)**2)*6*n1 #半成品拆解成本
            AA1=(1-q1)**3*(1-q4)/(1-G1*q1)**2*n1
            AA2=q4*(1-q1)**3/(1-G1*q1)**2*n1
            AA3=3*(1-q1)**2*(1-G1)*q1/(1-G1*q1)**2*n1
            AA4=3*(1-q1)*((1-G1)*q1)**2/(1-G1*q1)**2*n1
            AA5=(1-G1)**3*q1**3/(1-G1*q1)**2
            N=(1-G1*q1-(1-q4)*((1-q1)**3)/(1-G1*q1)**2)*n1 #更新数量
            # print(N)
            # print(q1)
            q=((((1-q1)**2)*(1-G1)*q1/(1-G1*q1)**2)+(1-q1)*(((1-G1)*q1)**2)/((1-G1*q1)**2)*2+(1-q1)*(((1-G1)*q1)**2)/(1-G1*q1)**2)/N #更新次品率
            
            return c21,c31,c22_1,c41_1,N,q,AA1,AA2,AA3,AA4,AA5

    def two_1(n1,q1,G1,H1,I1,q4=0.06):
        c21=G1*3*n1 #零件检测成本
        c31=(1-G1*q1)*8*n1 #半成品装配成本
        c22_1=H1*(1-G1*q1)*4*n1 #半成品检测成本
        c41_1=H1*I1*(1-G1*q1-(1-q4)*(1-q1)**2/(1-G1*q1))*6*n1#半成品拆解成本
        CC1=(1-q4)*(1-q1)**2/(1-G1*q1)*n1
        CC2=2*(1-q1)*q1*n1/(1-G1*q1)
        CC3=((1-G1)*q1)**2/(1-G1*q1)*n1
        CC4=q4*(1-q1)**2/(1-G1*q1)*n1
        N=(1-G1*q1-(1-q4)*((1-q1)**3)/(1-G1*q1)**2)*n1 #更新数量
        q=((1-q1)*q1*n1/(1-G1*q1)+((1-G1)*q1)**2/(1-G1*q1))/N#更新次品率
        return c21,c31,c22_1,c41_1,N,q,CC1,CC2,CC3,CC4
    A1,A2,A3,A4,A5=0,0,0,0,0 #用来记录半成品1的状态
    B1,B2,B3,B4,B5=0,0,0,0,0# 用来记录半成品2的状态
    O1,O2,O3,O4=0,0,0,0#用来记录半成品3的状态
    M1,M2,M3=100,100,100 #存储零件数量
    V1,V2,V3=0.06,0.06,0.06 #储存次品率
    W=0
    S=0
    pi2=0

    A=A1+A2+A3+A4+A5
    B=B1+B2+B3+B4+B5
    O=O1+O2+O3+O4

                                                                    
    lst1=[]#列表记录生成半成品1的决策过程
    G1=random.choice([0,1])
    H1=random.choice([0,1])
    I1=random.choice([0,1])
    lst1.append(G1)
    lst1.append(H1)
    lst1.append(I1)
    c21,c31,c22_1,c41_1,N,q,A1,A2,A3,A4,A5=three_1(M1,V1,G1,H1,I1,q4=0.06)
    S+=c21+c31+c22_1+c41_1
    A2*=(1-H1)
    A3*=(1-H1)
    A4*=(1-H1)
    A5*=(1-H1)
    n=0#设置n来指示阈值
    while H1*I1==1:
        if n>2:
            break
        else:
            if G1==1: #第一轮检测零件后第二轮不应该检测
                G1=0
            else:
                G1=random.choice([0,1])
        H1=random.choice([0,1])
        I1=random.choice([0,1])
        lst1.append(G1)
        lst1.append(H1)
        lst1.append(I1)
        c21,c31,c22_1,c41_1,N,q,AA1,AA2,AA3,AA4,AA5=three_1(N,q,G1,H1,I1,q4=0.06)
        S+=c21+c31+c22_1+c41_1
        A1+=AA1
        A2,A3,A4,A5=AA2,AA3,AA4,AA5
    #print(lst1)
    lst2=[]#列表记录生成半成品2的决策过程
    G2=random.choice([0,1])
    H2=random.choice([0,1])
    I2=random.choice([0,1])
    lst2.append(G2)
    lst2.append(H2)
    lst2.append(I2)
    c21,c31,c22_1,c41_1,N,q,B1,B2,B3,B4,B5=three_1(M2,V2,G2,H2,I2,q4=0.15)
    S+=c21+c31+c22_1+c41_1
    B2*=(1-H2)
    B3*=(1-H2)
    B4*=(1-H2)
    B5*=(1-H2)
    n=0#设置n来指示阈值
    while H2*I2==1:
        if n>2:
            break
        if G2==1: #第一轮检测零件后第二轮不应该检测
            G2=0
        else:
            G2=random.choice([0,1])
        H2=random.choice([0,1])
        I2=random.choice([0,1])
        lst2.append(G2)
        lst2.append(H2)
        lst2.append(I2)
        c21,c31,c22_1,c41_1,N,q,AA1,AA2,AA3,AA4,AA5=three_1(N,q,G2,H2,I2,q4=0.06)
        S+=c21+c31+c22_1+c41_1
        B1+=AA1
        B2,B3,B4,B5=AA2,AA3,AA4,AA5
        n=n+1

    lst3=[] #列表记录生成半成品3的决策过程
    G3=random.choice([0,1])
    H3=random.choice([0,1])
    I3=random.choice([0,1])
    lst3.append(G3)
    lst3.append(H3)
    lst3.append(I3)
    c21,c31,c22_1,c41_1,N,q,O1,O2,O3,O4=two_1(M3,V3,G3,H3,I3,q4=0.06)
    S+=c21+c31+c22_1+c41_1
    O2*=(1-H3)
    O3*=(1-H3)
    O4*=(1-H3)
    n=0#设置n来指示阈值
    while H3*I3==1:
        if n>2:
            break
        if G3==1: #第一轮检测零件后第二轮不应该检测
            G3=0
        else:
            G3=random.choice([0,1])
        H3=random.choice([0,1])
        I3=random.choice([0,1])
        lst3.append(G3)
        lst3.append(H3)
        lst3.append(I3)
        c21,c31,c22_1,c41_1,N,q,OO1,OO2,OO3,OO4=two_1(N,q,G3,H3,I3,q4=0.06)
        S+=c21+c31+c22_1+c41_1
        O1+=OO1
        O2,O3,O4=OO2,OO3,OO4
        n=n+1

    A=A1+A2+A3+A4+A5
    B=B1+B2+B3+B4+B5
    O=O1+O2+O3+O4
    # print(A)
    # print(A1)
    # print(A2)
    # print(B)
    # print(O)
    # print(S)
    # print(A1)
    # print(B1)
    # print(O1)
    # print(A)
    # print(B)
    # print(O)
    q2=0.06
    lst5=[]
    H7,H8,H9=random.choice([0,1]),random.choice([0,1]),random.choice([0,1])
    I7,I8,I9=random.choice([0,1]),random.choice([0,1]),random.choice([0,1])
    lst5.append(H7)
    lst5.append(I7)
    lst5.append(H8)
    lst5.append(I8)
    lst5.append(H9)
    lst5.append(I9)
    J=random.choice([0,1])
    K=random.choice([0,1])
    lst5.append(J)
    lst5.append(K)
    lst6=[]
    lst6.append(lst1)
    lst6.append(lst2)
    lst6.append(lst3)
    lst6.append(lst5)

    if A==min(A,B,O):#先按A的走
        # print(1)
        c32=A*8 #成品装配成本
        S+=c32
        c23=J*A*4 #成品检测成本
        S+=c23
        S=S+(1-J)*(A-(1-q2)*A1*B1*O1/B/O)*40 #成品的调换损失
        c42=K*(A-(1-q2)*A1*B1*O1/B/O)*10  #成品的拆解损失
        S=S+c42 

        M1=A-A1
        # print(A)
        # print(A1)
        # print(A5)
        M2=B-B1
        # print(B)
        # print(B1)
        # print(B4)
        M3=O-O1
        # print(O)
        # print(O1)
        # print(M1,end='*')
        # print(M2,end='*')
        # print(M3,end='*')
        V1=(A2/3+A3/3*2+A4)/(A-A1+q2*A1*B1*O1/B/O)
        V2=(B2/3+B3/3*2+B4)/(B-B1+q2*A1*B1*O1/B/O)
        V3=(O2/2+O3)/(O-O1+q2*A1*B1*O1/B/O)
        W=W+((1-q2)*A1*B1*O1/B/O)*200
        if K==1:
            S=S+H7*(A-(1-q2)*A1*B1*O1/B/O)*4+H8*(B-(1-q2)*A1*B1*O1/B/O)*4+H9*(O-(1-q2)*A1*B1*O1/B/O)*4
            S=S+H7*I7*(A-(1-q2)*A1*B1*O1/B/O)*6+I8*H8*(B-(1-q2)*A1*B1*O1/B/O)*6+I9*H9*(O-(1-q2)*A1*B1*O1/B/O)*6
        A1=A1-(1-q2)*A1*B1*O1/B/O #更新A1状态
        B1=B1-(1-q2)*A1*B1*O1/B/O #更新B1状态
        O1-=(1-q2)*A1*B1*O1/B/O #更新C1状态
    else:
        if O==min(A,B,O):
            # print(2)
            c32=O*8 #成品装配成本
            S=S+c32
            c23=J*O*4 #成品检测成本
            S=S+c23
            S=S+(1-J)*(O-(1-q2)*A1*B1*O1/B/A)*40 #成品的调换损失
            c42=K*(O-(1-q2)*A1*B1*O1/B/A)*10   #成品的拆解损失
            S=S+c42
            
            M1=A-A1
            M2=B-B1
            M3=O-O1
            # print(M1,end='*')
            # print(M2,end='*')
            # print(M3,end='*')
            V1=(A2/3+A3/3*2+A4)/(A-A1+q2*A1*B1*O1/B/A)
            V2=(B2/3+B3/3*2+B4)/(B-B1+q2*A1*B1*O1/B/A)
            V3=(O2/2+O3)/(O-O1+q2*A1*B1*O1/B/A)
            W=W+((1-q2)*A1*B1*O1/B/A)*200

            if K==1:
                S=S+H7*(A-(1-q2)*A1*B1*O1/B/A)*4+H8*(B-(1-q2)*A1*B1*O1/B/A)*4+H9*(O-(1-q2)*A1*B1*O1/B/A)*4
                S=S+H7*I7*(A-(1-q2)*A1*B1*O1/B/A)*6+I8*H8*(B-(1-q2)*A1*B1*O1/B/A)*6+I9*H9*(O-(1-q2)*A1*B1*O1/B/A)*6
            A1=A1-(1-q2)*A1*B1*O1/B/A
            B1=B1-(1-q2)*A1*B1*O1/B/A
            O1=O1-(1-q2)*A1*B1*O1/B/A
        else:
            # print(3)
            c32=B*8 #成品装配成本
            S=S+c32
            c23=J*B*4 #成品检测成本
            S=S+c23
            S=S+(1-J)*(B-(1-q2)*A1*B1*O1/A/O)*40 #成品的调换损失
            c42=K*(B-(1-q2)*A1*B1*O1/A/O)*10   #成品的拆解损失
            S=S+c42 

            M1=A-A1
            M2=B-B1
            M3=O-O1
            # print(M1,end='*')
            # print(M2,end='*')
            # print(M3,end='*')
            V1=(A2/3+A3/3*2+A4)/(A-A1+q2*A1*B1*O1/A/O)
            V2=(B2/3+B3/3*2+B4)/(B-B1+q2*A1*B1*O1/A/O)
            V3=(O2/2+O3)/(O-O1+q2*A1*B1*O1/A/O) 
            W=W+((1-q2)*A1*B1*O1/O/A)*200
            if K==1:
                S=S+H7*(A-(1-q2)*A1*B1*O1/O/A)*4+H8*(B-(1-q2)*A1*B1*O1/O/A)*4+H9*(O-(1-q2)*A1*B1*O1/O/A)*4
                S=S+H7*I7*(A-(1-q2)*A1*B1*O1/O/A)*6+I8*H8*(B-(1-q2)*A1*B1*O1/O/A)*6+I9*H9*(O-(1-q2)*A1*B1*O1/O/A)*6
            A1=A1-(1-q2)*A1*B1*O1/A/O #更新A1状态
            B1=B1-(1-q2)*A1*B1*O1/A/O #更新B1状态
            O1=O1-(1-q2)*A1*B1*O1/A/O #更新C1状态
    pi=pii=W-S
    lst4=[]
    lst4.append(pi)
    # print(S)
    # print(W)
    # print(A1)
    # print(A2)
    # print(A)
    ##print(pi)
    # print(' ')
    # print(K)
    lst7,lst8,lst9,lst13=['lst7'],['lst8'],['lst9'],['lst13']
    m=0
    while K==1:
       
        W,S=0,0
        if H7*I7==1 and (H1!=1 or I1!=0) :
            # print(4)
            G1=random.choice([0,1])
            H1=random.choice([0,1])
            I1=random.choice([0,1])
            lst7.append(G1)
            lst7.append(H1)
            lst7.append(I1)
            c21,c31,c22_1,c41_1,N,q,A1_,A2,A3,A4,A5=three_1(M1,V1,G1,H1,I1,q4=0.06)
            S+=c21+c31+c22_1+c41_1
            A1=A1+A1_
            A2*=(1-H1)
            A3*=(1-H1)
            A4*=(1-H1)
            A5*=(1-H1)
            n=0#设置n来指示阈值
            while H1*I1==1:
                if n>2:
                    break
                else:
                    if G1==1: #第一轮检测零件后第二轮不应该检测
                        G1=0
                    else:
                        G1=random.choice([0,1])
                H1=random.choice([0,1])
                I1=random.choice([0,1])
                lst7.append(G1)
                lst7.append(H1)
                lst7.append(I1)
                c21,c31,c22_1,c41_1,N,q,AA1,AA2,AA3,AA4,AA5=three_1(N,q,G1,H1,I1,q4=0.06)
                S+=c21+c31+c22_1+c41_1
                A1+=AA1
                A2,A3,A4,A5=AA2,AA3,AA4,AA5
            lst7.append('done')
        if H8*I8==1 and (H2!=1 or I2!=0):
            # print(5)
            G2=random.choice([0,1])
            H2=random.choice([0,1])
            I2=random.choice([0,1])
            lst8.append(G2)
            lst8.append(H2)
            lst8.append(I2)
            c21,c31,c22_1,c41_1,N,q,B1_,B2,B3,B4,B5=three_1(M2,V2,G2,H2,I2,q4=0.06)
            S+=c21+c31+c22_1+c41_1
            B1=B1+B1_
            B2*=(1-H2)
            B3*=(1-H2)
            B4*=(1-H2)
            B5*=(1-H2)
            n=0#设置n来指示阈值
            while H2*I2==1:
                if n>2:
                    break
                else:
                    if G2==1: #第一轮检测零件后第二轮不应该检测
                        G2=0
                    else:
                        G2=random.choice([0,1])
                H2=random.choice([0,1])
                I2=random.choice([0,1])
                lst8.append(G2)
                lst8.append(H2)
                lst8.append(I2)
                c21,c31,c22_1,c41_1,N,q,AA1,AA2,AA3,AA4,AA5=three_1(N,q,G2,H2,I2,q4=0.06)
                S+=c21+c31+c22_1+c41_1
                B1+=AA1
                B2,B3,B4,B5=AA2,AA3,AA4,AA5
            lst8.append('done')
        if H9*I9==1 and (H3!=1 or I3!=0):
            # print(6)
            G3=random.choice([0,1])
            H3=random.choice([0,1])
            I3=random.choice([0,1])
            lst9.append(G3)
            lst9.append(H3)
            lst9.append(I3)
            c21,c31,c22_1,c41_1,N,q,O1_,O2,O3,O4=two_1(M3,V3,G3,H3,I3,q4=0.06)
            S+=c21+c31+c22_1+c41_1
            O1=O1+O1_
            O2*=(1-H3)
            O3*=(1-H3)
            O4*=(1-H3)
            n=0#设置n来指示阈值
            while H3*I3==1:
                if n>2:
                    break
                else:
                    if G3==1: #第一轮检测零件后第二轮不应该检测
                        G3=0
                    else:
                        G3=random.choice([0,1])
                H3=random.choice([0,1])
                I3=random.choice([0,1])
                lst9.append(G3)
                lst9.append(H3)
                lst9.append(I3)
                c21,c31,c22_1,c41_1,N,q,AA1,AA2,AA3,AA4=two_1(N,q,G3,H3,I3,q4=0.06)
                S+=c21+c31+c22_1+c41_1
                O1+=AA1
                O2,O3,O4=AA2,AA3,AA4
            lst9.append('done')
        
        H7,H8,H9=random.choice([0,1]),random.choice([0,1]),random.choice([0,1])
        I7,I8,I9=random.choice([0,1]),random.choice([0,1]),random.choice([0,1])
        J=random.choice([0,1])
        K=random.choice([0,1])
     
        A=A1+A2+A3+A4+A5
        B=B1+B2+B3+B4+B5
        O=O1+O2+O3+O4
        if A==min(A,B,O) :#先按A的走
            # print(1)
            c32=A*8 #成品装配成本
            S+=c32
            c23=J*A*4 #成品检测成本
            S+=c23
            S=S+(1-J)*(A-(1-q2)*A1*B1*O1/B/O)*40 #成品的调换损失
            c42=K*(A-(1-q2)*A1*B1*O1/B/O)*10  #成品的拆解损失
            S=S+c42 
        
            M1=A-A1
            M2=B-B1
            M3=O-O1
            V1=(A2/3+A3/3*2+A4)/(A-A1+q2*A1*B1*O1/B/O)
            V2=(B2/3+B3/3*2+B4)/(B-B1+q2*A1*B1*O1/B/O)
            V3=(O2/2+O3)/(O-O1+q2*A1*B1*O1/B/O)
            W=((1-q2)*A1*B1*O1/B/O)*200
    
            if K==1:
                S=S+H7*(A-(1-q2)*A1*B1*O1/B/O)*4+H8*(B-(1-q2)*A1*B1*O1/B/O)*4+H9*(O-(1-q2)*A1*B1*O1/B/O)*4
                S=S+H7*I7*(A-(1-q2)*A1*B1*O1/B/O)*6+I8*H8*(B-(1-q2)*A1*B1*O1/B/O)*6+I9*H9*(O-(1-q2)*A1*B1*O1/B/O)*6
            A1=A1-(1-q2)*A1*B1*O1/B/O #更新A1状态
            B1=B1-(1-q2)*A1*B1*O1/B/O #更新B1状态
            O1-=(1-q2)*A1*B1*O1/B/O #更新O1状态
        else:
            if O==min(A,B,O):
                # print(2)
                c32=O*8 #成品装配成本
                S=S+c32
                c23=J*O*4 #成品检测成本
                S=S+c23
                S=S+(1-J)*(O-(1-q2)*A1*B1*O1/B/A)*40 #成品的调换损失
                c42=K*(O-(1-q2)*A1*B1*O1/B/A)*10   #成品的拆解损失
                S=S+c42
                
                M1=A-A1
                M2=B-B1
                M3=O-O1
                V1=(A2/3+A3/3*2+A4)/(A-A1+q2*A1*B1*O1/B/A)
                V2=(B2/3+B3/3*2+B4)/(B-B1+q2*A1*B1*O1/B/A)
                V3=(O2/2+O3)/(O-O1+q2*A1*B1*O1/B/A)
                W=((1-q2)*A1*B1*O1/B/A)*200
            
                if K==1:
                    S=S+H7*(A-(1-q2)*A1*B1*O1/B/A)*4+H8*(B-(1-q2)*A1*B1*O1/B/A)*4+H9*(O-(1-q2)*A1*B1*O1/B/A)*4
                    S=S+H7*I7*(A-(1-q2)*A1*B1*O1/B/A)*6+I8*H8*(B-(1-q2)*A1*B1*O1/B/A)*6+I9*H9*(O-(1-q2)*A1*B1*O1/B/A)*6
                A1=A1-(1-q2)*A1*B1*O1/B/A
                B1=B1-(1-q2)*A1*B1*O1/B/A
                O1=O1-(1-q2)*A1*B1*O1/B/A
            else:
                # print(3)
                c32=B*8 #成品装配成本
                S=S+c32
                c23=J*B*4 #成品检测成本
                S=S+c23
                S=S+(1-J)*(B-(1-q2)*A1*B1*O1/A/O)*40 #成品的调换损失
                c42=K*(B-(1-q2)*A1*B1*O1/A/O)*10   #成品的拆解损失
                S=S+c42 
            
                M1=A-A1
                M2=B-B1
                M3=O-O1

                V1=(A2/3+A3/3*2+A4)/(A-A1+q2*A1*B1*O1/A/O)
                V2=(B2/3+B3/3*2+B4)/(B-B1+q2*A1*B1*O1/A/O)
                V3=(O2/2+O3)/(O-O1+q2*A1*B1*O1/A/O)
                W=((1-q2)*A1*B1*O1/O/A)*200
                if K==1:
                    S=S+H7*(A-(1-q2)*A1*B1*O1/O/A)*4+H8*(B-(1-q2)*A1*B1*O1/O/A)*4+H9*(O-(1-q2)*A1*B1*O1/O/A)*4
                    S=S+H7*I7*(A-(1-q2)*A1*B1*O1/O/A)*6+I8*H8*(B-(1-q2)*A1*B1*O1/O/A)*6+I9*H9*(O-(1-q2)*A1*B1*O1/O/A)*6
                A1=A1-(1-q2)*A1*B1*O1/A/O #更新A1状态
                B1=B1-(1-q2)*A1*B1*O1/A/O #更新B1状态
                O1=O1-(1-q2)*A1*B1*O1/A/O #更新C1状态
        pi=W-S
        
        if pi>0:
            # print(W,end='*')
            # print(S,end='*')
            if m<2:
                lst131.append(W)
                lst131.append(S)
                lst4.append(pi)
                lst13.append(H7)
                lst13.append(I7)
                lst13.append(H8)
                lst13.append(I8)
                lst13.append(H9)
                lst13.append(I9)
                lst13.append(J)
                lst13.append(K)
            
            else:
                break 
        else:
            break

        m=m+1
       
    lst151.append(sum(lst4))
    lst141.append(lst131)
    if number<sum(lst4):
        number=sum(lst4)
    lst10=[]
    lst10.append(lst6+lst7+lst8+lst9+lst13)
    lst11.append(lst10)
    lst7,lst8,lst9,lst13=['lst7'],['lst8'],['lst9'],['lst13']
print(number)
print(lst141[lst151.index(number)])
print(lst11[lst151.index(number)])

import random
number=0
lst11=[]#存储决策
lst12=[]#存储利润
lst141,lst151=[],[]
for i in range(10000):
    lst131=[]
    
    # try:
    def three_1(n1,q1,G1,H1,I1,q4=0.06):
            #print(n1)
            c21=G1*(n1+n1+2*n1) #零件检测成本
            c31=(1-G1*q1)*8*n1 #半成品装配成本
            c22_1=H1*(1-G1*q1)*4*n1 #半成品检测成本
            c41_1=H1*I1*(1-G1*q1-((1-q1)**3)*(1-q4)/(1-G1*q1)**2)*6*n1 #半成品拆解成本
            AA1=(1-q1)**3*(1-q4)/(1-G1*q1)**2*n1
            AA2=q4*(1-q1)**3/(1-G1*q1)**2*n1
            AA3=3*(1-q1)**2*(1-G1)*q1/(1-G1*q1)**2*n1
            AA4=3*(1-q1)*((1-G1)*q1)**2/(1-G1*q1)**2*n1
            AA5=(1-G1)**3*q1**3/(1-G1*q1)**2
            N=(1-G1*q1-(1-q4)*((1-q1)**3)/(1-G1*q1)**2)*n1 #更新数量
            # print(N)
            # print(q1)
            q=((((1-q1)**2)*(1-G1)*q1/(1-G1*q1)**2)+(1-q1)*(((1-G1)*q1)**2)/((1-G1*q1)**2)*2+(1-q1)*(((1-G1)*q1)**2)/(1-G1*q1)**2)/N #更新次品率
            
            return c21,c31,c22_1,c41_1,N,q,AA1,AA2,AA3,AA4,AA5

    def two_1(n1,q1,G1,H1,I1,q4=0.06):
        c21=G1*3*n1 #零件检测成本
        c31=(1-G1*q1)*8*n1 #半成品装配成本
        c22_1=H1*(1-G1*q1)*4*n1 #半成品检测成本
        c41_1=H1*I1*(1-G1*q1-(1-q4)*(1-q1)**2/(1-G1*q1))*6*n1#半成品拆解成本
        CC1=(1-q4)*(1-q1)**2/(1-G1*q1)*n1
        CC2=2*(1-q1)*q1*n1/(1-G1*q1)
        CC3=((1-G1)*q1)**2/(1-G1*q1)*n1
        CC4=q4*(1-q1)**2/(1-G1*q1)*n1
        N=(1-G1*q1-(1-q4)*((1-q1)**3)/(1-G1*q1)**2)*n1 #更新数量
        q=((1-q1)*q1*n1/(1-G1*q1)+((1-G1)*q1)**2/(1-G1*q1))/N#更新次品率
        return c21,c31,c22_1,c41_1,N,q,CC1,CC2,CC3,CC4
    A1,A2,A3,A4,A5=0,0,0,0,0 #用来记录半成品1的状态
    B1,B2,B3,B4,B5=0,0,0,0,0# 用来记录半成品2的状态
    O1,O2,O3,O4=0,0,0,0#用来记录半成品3的状态
    M1,M2,M3=100,100,100 #存储零件数量
    V1,V2,V3=0.1,0.1,0.1 #储存次品率
    W=0
    S=0
    pi2=0

    A=A1+A2+A3+A4+A5
    B=B1+B2+B3+B4+B5
    O=O1+O2+O3+O4

                                                                    
    lst1=[]#列表记录生成半成品1的决策过程
    G1=random.choice([0,1])
    H1=random.choice([0,1])
    I1=random.choice([0,1])
    lst1.append(G1)
    lst1.append(H1)
    lst1.append(I1)
    c21,c31,c22_1,c41_1,N,q,A1,A2,A3,A4,A5=three_1(M1,V1,G1,H1,I1,q4=0.06)
    S+=c21+c31+c22_1+c41_1
    A2*=(1-H1)
    A3*=(1-H1)
    A4*=(1-H1)
    A5*=(1-H1)
    n=0#设置n来指示阈值
    while H1*I1==1:
        if n>2:
            break
        else:
            if G1==1: #第一轮检测零件后第二轮不应该检测
                G1=0
            else:
                G1=random.choice([0,1])
        H1=random.choice([0,1])
        I1=random.choice([0,1])
        lst1.append(G1)
        lst1.append(H1)
        lst1.append(I1)
        c21,c31,c22_1,c41_1,N,q,AA1,AA2,AA3,AA4,AA5=three_1(N,q,G1,H1,I1,q4=0.06)
        S+=c21+c31+c22_1+c41_1
        A1+=AA1
        A2,A3,A4,A5=AA2,AA3,AA4,AA5
    #print(lst1)
    lst2=[]#列表记录生成半成品2的决策过程
    G2=random.choice([0,1])
    H2=random.choice([0,1])
    I2=random.choice([0,1])
    lst2.append(G2)
    lst2.append(H2)
    lst2.append(I2)
    c21,c31,c22_1,c41_1,N,q,B1,B2,B3,B4,B5=three_1(M2,V2,G2,H2,I2,q4=0.06)
    S+=c21+c31+c22_1+c41_1
    B2*=(1-H2)
    B3*=(1-H2)
    B4*=(1-H2)
    B5*=(1-H2)
    n=0#设置n来指示阈值
    while H2*I2==1:
        if n>2:
            break
        if G2==1: #第一轮检测零件后第二轮不应该检测
            G2=0
        else:
            G2=random.choice([0,1])
        H2=random.choice([0,1])
        I2=random.choice([0,1])
        lst2.append(G2)
        lst2.append(H2)
        lst2.append(I2)
        c21,c31,c22_1,c41_1,N,q,AA1,AA2,AA3,AA4,AA5=three_1(N,q,G2,H2,I2,q4=0.06)
        S+=c21+c31+c22_1+c41_1
        B1+=AA1
        B2,B3,B4,B5=AA2,AA3,AA4,AA5
        n=n+1

    lst3=[] #列表记录生成半成品3的决策过程
    G3=random.choice([0,1])
    H3=random.choice([0,1])
    I3=random.choice([0,1])
    lst3.append(G3)
    lst3.append(H3)
    lst3.append(I3)
    c21,c31,c22_1,c41_1,N,q,O1,O2,O3,O4=two_1(M3,V3,G3,H3,I3,q4=0.06)
    S+=c21+c31+c22_1+c41_1
    O2*=(1-H3)
    O3*=(1-H3)
    O4*=(1-H3)
    n=0#设置n来指示阈值
    while H3*I3==1:
        if n>2:
            break
        if G3==1: #第一轮检测零件后第二轮不应该检测
            G3=0
        else:
            G3=random.choice([0,1])
        H3=random.choice([0,1])
        I3=random.choice([0,1])
        lst3.append(G3)
        lst3.append(H3)
        lst3.append(I3)
        c21,c31,c22_1,c41_1,N,q,OO1,OO2,OO3,OO4=two_1(N,q,G3,H3,I3,q4=0.06)
        S+=c21+c31+c22_1+c41_1
        O1+=OO1
        O2,O3,O4=OO2,OO3,OO4
        n=n+1

    A=A1+A2+A3+A4+A5
    B=B1+B2+B3+B4+B5
    O=O1+O2+O3+O4
    # print(A)
    # print(A1)
    # print(A2)
    # print(B)
    # print(O)
    # print(S)
    # print(A1)
    # print(B1)
    # print(O1)
    # print(A)
    # print(B)
    # print(O)
    q2=0.06
    lst5=[]
    #H7,H8,H9=random.choice([0,1]),random.choice([0,1]),random.choice([0,1])
    I7,I8,I9=random.choice([0,1]),random.choice([0,1]),random.choice([0,1])
    if H1==1 and I1==0:
        H7=0
    else: 
        H7=random.choice([0,1])
    if H2==1 and I2==0:
        H8=0
    else:
        H8=random.choice([0,1])
    if H3==1 and I3==0:
        H9=0
    else:
        H9=random.choice([0,1])
    lst5.append(H7)
    lst5.append(I7)
    lst5.append(H8)
    lst5.append(I8)
    lst5.append(H9)
    lst5.append(I9)
    J=random.choice([0,1])
    K=random.choice([0,1])
    lst5.append(J)
    lst5.append(K)
    lst6=[]
    lst6.append(lst1)
    lst6.append(lst2)
    lst6.append(lst3)
    lst6.append(lst5)

    if A==min(A,B,O):#先按A的走
        # print(1)
        c32=A*8 #成品装配成本
        S+=c32
        c23=J*A*4 #成品检测成本
        S+=c23
        S=S+(1-J)*(A-(1-q2)*A1*B1*O1/B/O)*40 #成品的调换损失
        c42=K*(A-(1-q2)*A1*B1*O1/B/O)*10  #成品的拆解损失
        S=S+c42 

        M1=A-A1
        # print(A)
        # print(A1)
        # print(A5)
        M2=B-B1
        # print(B)
        # print(B1)
        # print(B4)
        M3=O-O1
        # print(O)
        # print(O1)
        # print(M1,end='*')
        # print(M2,end='*')
        # print(M3,end='*')
        V1=(A2/3+A3/3*2+A4)/(A-A1+q2*A1*B1*O1/B/O)
        V2=(B2/3+B3/3*2+B4)/(B-B1+q2*A1*B1*O1/B/O)
        V3=(O2/2+O3)/(O-O1+q2*A1*B1*O1/B/O)
        W=W+((1-q2)*A1*B1*O1/B/O)*200
        A1=A1-(1-q2)*A1*B1*O1/B/O #更新A1状态
        B1=B1-(1-q2)*A1*B1*O1/B/O #更新B1状态
        O1-=(1-q2)*A1*B1*O1/B/O #更新C1状态

        if K==1:
            S=S+H7*(A-(1-q2)*A1*B1*O1/B/O)*4+H8*(B-(1-q2)*A1*B1*O1/B/O)*4+H9*(O-(1-q2)*A1*B1*O1/B/O)*4
            S=S+H7*I7*(A-(1-q2)*A1*B1*O1/B/O)*6+I8*H8*(B-(1-q2)*A1*B1*O1/B/O)*6+I9*H9*(O-(1-q2)*A1*B1*O1/B/O)*6
      
            if H7==1 and I7==0:
                A=A1
                A2,A3,A4,A5=0,0,0,0
            if H8==1 and I8==0:
                B=B1
                B2,B3,B4,B5=0,0,0,0
            if H9==1 and I9==0:
                O=O1
                O2,O3,O4=0,0,0
    else:
        if O==min(A,B,O):
            # print(2)
            c32=O*8 #成品装配成本
            S=S+c32
            c23=J*O*4 #成品检测成本
            S=S+c23
            S=S+(1-J)*(O-(1-q2)*A1*B1*O1/B/A)*40 #成品的调换损失
            c42=K*(O-(1-q2)*A1*B1*O1/B/A)*10   #成品的拆解损失
            S=S+c42
            
            M1=A-A1
            M2=B-B1
            M3=O-O1
            # print(M1,end='*')
            # print(M2,end='*')
            # print(M3,end='*')
            V1=(A2/3+A3/3*2+A4)/(A-A1+q2*A1*B1*O1/B/A)
            V2=(B2/3+B3/3*2+B4)/(B-B1+q2*A1*B1*O1/B/A)
            V3=(O2/2+O3)/(O-O1+q2*A1*B1*O1/B/A)
            W=W+((1-q2)*A1*B1*O1/B/A)*200
            A1=A1-(1-q2)*A1*B1*O1/B/A
            B1=B1-(1-q2)*A1*B1*O1/B/A
            O1=O1-(1-q2)*A1*B1*O1/B/A
            
            if K==1:
                S=S+H7*(A-(1-q2)*A1*B1*O1/B/A)*4+H8*(B-(1-q2)*A1*B1*O1/B/A)*4+H9*(O-(1-q2)*A1*B1*O1/B/A)*4
                S=S+H7*I7*(A-(1-q2)*A1*B1*O1/B/A)*6+I8*H8*(B-(1-q2)*A1*B1*O1/B/A)*6+I9*H9*(O-(1-q2)*A1*B1*O1/B/A)*6
           
                if H7==1 and I7==0:
                    A=A1
                    A2,A3,A4,A5=0,0,0,0
                if H8==1 and I8==0:
                    B=B1
                    B2,B3,B4,B5=0,0,0,0
                if H9==1 and I9==0:
                    O=O1
                    O2,O3,O4=0,0,0
        else:
            # print(3)
            c32=B*8 #成品装配成本
            S=S+c32
            c23=J*B*4 #成品检测成本
            S=S+c23
            S=S+(1-J)*(B-(1-q2)*A1*B1*O1/A/O)*40 #成品的调换损失
            c42=K*(B-(1-q2)*A1*B1*O1/A/O)*10   #成品的拆解损失
            S=S+c42 

            M1=A-A1
            M2=B-B1
            M3=O-O1
            # print(M1,end='*')
            # print(M2,end='*')
            # print(M3,end='*')
            V1=(A2/3+A3/3*2+A4)/(A-A1+q2*A1*B1*O1/A/O)
            V2=(B2/3+B3/3*2+B4)/(B-B1+q2*A1*B1*O1/A/O)
            V3=(O2/2+O3)/(O-O1+q2*A1*B1*O1/A/O) 
            W=W+((1-q2)*A1*B1*O1/O/A)*200
            A1=A1-(1-q2)*A1*B1*O1/A/O #更新A1状态
            B1=B1-(1-q2)*A1*B1*O1/A/O #更新B1状态
            O1=O1-(1-q2)*A1*B1*O1/A/O #更新C1状态
            if K==1:
                S=S+H7*(A-(1-q2)*A1*B1*O1/O/A)*4+H8*(B-(1-q2)*A1*B1*O1/O/A)*4+H9*(O-(1-q2)*A1*B1*O1/O/A)*4
                S=S+H7*I7*(A-(1-q2)*A1*B1*O1/O/A)*6+I8*H8*(B-(1-q2)*A1*B1*O1/O/A)*6+I9*H9*(O-(1-q2)*A1*B1*O1/O/A)*6
            
                if H7==1 and I7==0:
                    A=A1
                    A2,A3,A4,A5=0,0,0,0
                if H8==1 and I8==0:
                    B=B1
                    B2,B3,B4,B5=0,0,0,0
                if H9==1 and I9==0:
                    O=O1
                    O2,O3,O4=0,0,0
    pi=pii=W-S
    lst4=[]
    lst4.append(pi)
    # print(S)
    # print(W)
    # print(A1)
    # print(A2)
    # print(A)
    ##print(pi)
    # print(' ')
    # print(K)
    lst7,lst8,lst9,lst13=['lst7'],['lst8'],['lst9'],['lst13']
    m=0
    while K==1:
        W,S=0,0
        if H7*I7==1 and M1!=0 :
            #if (H1!=1 or I1!=0):
            # print(4)
            G1=random.choice([0,1])
            H1=random.choice([0,1])
            I1=random.choice([0,1])
            lst7.append(G1)
            lst7.append(H1)
            lst7.append(I1)
            c21,c31,c22_1,c41_1,N,q,A1_,A2,A3,A4,A5=three_1(M1,V1,G1,H1,I1,q4=0.06)
            S+=c21+c31+c22_1+c41_1
            A1=A1+A1_
            A2*=(1-H1)
            A3*=(1-H1)
            A4*=(1-H1)
            A5*=(1-H1)
            n=0#设置n来指示阈值
            while H1*I1==1:
                if n>2:
                    break
                else:
                    if G1==1: #第一轮检测零件后第二轮不应该检测
                        G1=0
                    else:
                        G1=random.choice([0,1])
                H1=random.choice([0,1])
                I1=random.choice([0,1])
                lst7.append(G1)
                lst7.append(H1)
                lst7.append(I1)
                c21,c31,c22_1,c41_1,N,q,AA1,AA2,AA3,AA4,AA5=three_1(N,q,G1,H1,I1,q4=0.06)
                S+=c21+c31+c22_1+c41_1
                A1+=AA1
                A2,A3,A4,A5=AA2,AA3,AA4,AA5
        lst7.append('done')
                
                
        if H8*I8==1 and M2!=0:
            # print(5)
            G2=random.choice([0,1])
            H2=random.choice([0,1])
            I2=random.choice([0,1])
            lst8.append(G2)
            lst8.append(H2)
            lst8.append(I2)
            c21,c31,c22_1,c41_1,N,q,B1_,B2,B3,B4,B5=three_1(M2,V2,G2,H2,I2,q4=0.06)
            S+=c21+c31+c22_1+c41_1
            B1=B1+B1_
            B2*=(1-H2)
            B3*=(1-H2)
            B4*=(1-H2)
            B5*=(1-H2)
            n=0#设置n来指示阈值
            while H2*I2==1:
                if n>2:
                    break
                else:
                    if G2==1: #第一轮检测零件后第二轮不应该检测
                        G2=0
                    else:
                        G2=random.choice([0,1])
                H2=random.choice([0,1])
                I2=random.choice([0,1])
                lst8.append(G2)
                lst8.append(H2)
                lst8.append(I2)
                c21,c31,c22_1,c41_1,N,q,AA1,AA2,AA3,AA4,AA5=three_1(N,q,G2,H2,I2,q4=0.06)
                S+=c21+c31+c22_1+c41_1
                B1+=AA1
                B2,B3,B4,B5=AA2,AA3,AA4,AA5
            lst8.append('done')
        if H9*I9==1 and M3!=0:
            # print(6)
            G3=random.choice([0,1])
            H3=random.choice([0,1])
            I3=random.choice([0,1])
            lst9.append(G3)
            lst9.append(H3)
            lst9.append(I3)
            c21,c31,c22_1,c41_1,N,q,O1_,O2,O3,O4=two_1(M3,V3,G3,H3,I3,q4=0.06)
            S+=c21+c31+c22_1+c41_1
            O1=O1+O1_
            O2*=(1-H3)
            O3*=(1-H3)
            O4*=(1-H3)
            n=0#设置n来指示阈值
            while H3*I3==1:
                if n>2:
                    break
                else:
                    if G3==1: #第一轮检测零件后第二轮不应该检测
                        G3=0
                    else:
                        G3=random.choice([0,1])
                H3=random.choice([0,1])
                I3=random.choice([0,1])
                lst9.append(G3)
                lst9.append(H3)
                lst9.append(I3)
                c21,c31,c22_1,c41_1,N,q,AA1,AA2,AA3,AA4=two_1(N,q,G3,H3,I3,q4=0.06)
                S+=c21+c31+c22_1+c41_1
                O1+=AA1
                O2,O3,O4=AA2,AA3,AA4
            lst9.append('done')
        
        #H7,H8,H9=random.choice([0,1]),random.choice([0,1]),random.choice([0,1])
        I7,I8,I9=random.choice([0,1]),random.choice([0,1]),random.choice([0,1])
        J=random.choice([0,1])
        K=random.choice([0,1])
        if H1==1 and I1==0:
            H7=0
        else: 
            H7=random.choice([0,1])
        if H2==1 and I2==0:
            H8=0
        else:
            H8=random.choice([0,1])
        if H3==1 and I3==0:
            H9=0
        else:
            H9=random.choice([0,1])
        A=A1+A2+A3+A4+A5
        B=B1+B2+B3+B4+B5
        O=O1+O2+O3+O4
        if A==min(A,B,O) :#先按A的走
            # print(1)
            c32=A*8 #成品装配成本
            S+=c32
            c23=J*A*4 #成品检测成本
            S+=c23
            S=S+(1-J)*(A-(1-q2)*A1*B1*O1/B/O)*40 #成品的调换损失
            c42=K*(A-(1-q2)*A1*B1*O1/B/O)*10  #成品的拆解损失
            S=S+c42 
        
            M1=A-A1
            M2=B-B1
            M3=O-O1
            V1=(A2/3+A3/3*2+A4)/(A-A1+q2*A1*B1*O1/B/O)
            V2=(B2/3+B3/3*2+B4)/(B-B1+q2*A1*B1*O1/B/O)
            V3=(O2/2+O3)/(O-O1+q2*A1*B1*O1/B/O)
            W=((1-q2)*A1*B1*O1/B/O)*200
            A1=A1-(1-q2)*A1*B1*O1/B/O #更新A1状态
            B1=B1-(1-q2)*A1*B1*O1/B/O #更新B1状态
            O1-=(1-q2)*A1*B1*O1/B/O #更新O1状态
            if K==1:
                S=S+H7*(A-(1-q2)*A1*B1*O1/B/O)*4+H8*(B-(1-q2)*A1*B1*O1/B/O)*4+H9*(O-(1-q2)*A1*B1*O1/B/O)*4
                S=S+H7*I7*(A-(1-q2)*A1*B1*O1/B/O)*6+I8*H8*(B-(1-q2)*A1*B1*O1/B/O)*6+I9*H9*(O-(1-q2)*A1*B1*O1/B/O)*6
           
                if H7==1 and I7==0:
                    A=A1
                    A2,A3,A4,A5=0,0,0,0
                if H8==1 and I8==0:
                    B=B1
                    B2,B3,B4,B5=0,0,0,0
                if H9==1 and I9==0:
                    O=O1
                    O2,O3,O4=0,0,0
            
        else:
            if O==min(A,B,O):
                # print(2)
                c32=O*8 #成品装配成本
                S=S+c32
                c23=J*O*4 #成品检测成本
                S=S+c23
                S=S+(1-J)*(O-(1-q2)*A1*B1*O1/B/A)*40 #成品的调换损失
                c42=K*(O-(1-q2)*A1*B1*O1/B/A)*10   #成品的拆解损失
                S=S+c42
                
                M1=A-A1
                M2=B-B1
                M3=O-O1
                V1=(A2/3+A3/3*2+A4)/(A-A1+q2*A1*B1*O1/B/A)
                V2=(B2/3+B3/3*2+B4)/(B-B1+q2*A1*B1*O1/B/A)
                V3=(O2/2+O3)/(O-O1+q2*A1*B1*O1/B/A)
                W=((1-q2)*A1*B1*O1/B/A)*200
                A1=A1-(1-q2)*A1*B1*O1/B/A
                B1=B1-(1-q2)*A1*B1*O1/B/A
                O1=O1-(1-q2)*A1*B1*O1/B/A

                if K==1:
                    S=S+H7*(A-(1-q2)*A1*B1*O1/B/A)*4+H8*(B-(1-q2)*A1*B1*O1/B/A)*4+H9*(O-(1-q2)*A1*B1*O1/B/A)*4
                    S=S+H7*I7*(A-(1-q2)*A1*B1*O1/B/A)*6+I8*H8*(B-(1-q2)*A1*B1*O1/B/A)*6+I9*H9*(O-(1-q2)*A1*B1*O1/B/A)*6
               
                    if H7==1 and I7==0:
                            A=A1
                            A2,A3,A4,A5=0,0,0,0
                    if H8==1 and I8==0:
                        B=B1
                        B2,B3,B4,B5=0,0,0,0
                    if H9==1 and I9==0:
                        O=O1
                        O2,O3,O4=0,0,0
            else:
                # print(3)
                c32=B*8 #成品装配成本
                S=S+c32
                c23=J*B*4 #成品检测成本
                S=S+c23
                S=S+(1-J)*(B-(1-q2)*A1*B1*O1/A/O)*40 #成品的调换损失
                c42=K*(B-(1-q2)*A1*B1*O1/A/O)*10   #成品的拆解损失
                S=S+c42 
            
                M1=A-A1
                M2=B-B1
                M3=O-O1

                V1=(A2/3+A3/3*2+A4)/(A-A1+q2*A1*B1*O1/A/O)
                V2=(B2/3+B3/3*2+B4)/(B-B1+q2*A1*B1*O1/A/O)
                V3=(O2/2+O3)/(O-O1+q2*A1*B1*O1/A/O)
                W=((1-q2)*A1*B1*O1/O/A)*200
                A1=A1-(1-q2)*A1*B1*O1/A/O #更新A1状态
                B1=B1-(1-q2)*A1*B1*O1/A/O #更新B1状态
                O1=O1-(1-q2)*A1*B1*O1/A/O #更新C1状态
                if K==1:
                    S=S+H7*(A-(1-q2)*A1*B1*O1/O/A)*4+H8*(B-(1-q2)*A1*B1*O1/O/A)*4+H9*(O-(1-q2)*A1*B1*O1/O/A)*4
                    S=S+H7*I7*(A-(1-q2)*A1*B1*O1/O/A)*6+I8*H8*(B-(1-q2)*A1*B1*O1/O/A)*6+I9*H9*(O-(1-q2)*A1*B1*O1/O/A)*6
                
                    if H7==1 and I7==0:
                        A=A1
                        A2,A3,A4,A5=0,0,0,0
                    if H8==1 and I8==0:
                        B=B1
                        B2,B3,B4,B5=0,0,0,0
                    if H9==1 and I9==0:
                        O=O1
                        O2,O3,O4=0,0,0
        pi=W-S
        
        if S>0 and pi>0 and W<10000:
            # print(W,end='*')
            # print(S,end='*')
            if m<2:
                lst131.append(W)
                lst131.append(S)
                lst4.append(pi)
                lst13.append(H7)
                lst13.append(I7)
                lst13.append(H8)
                lst13.append(I8)
                lst13.append(H9)
                lst13.append(I9)
                lst13.append(J)
                lst13.append(K)
            else:
                break 
        else:
            break

        m=m+1
    lst151.append(sum(lst4))
    lst141.append(lst131)
    if number<sum(lst4):
        number=sum(lst4)
    lst10=[]
    lst10.append(lst6+lst7+lst8+lst9+lst13)
    lst11.append(lst10)

print(number)
print(lst141[lst151.index(number)])
print(lst11[lst151.index(number)])