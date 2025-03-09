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

q1,q2,q3,a1,a2,a3,a4,a5,a6,a7,a8,a9=0.1,0.1,0.1,4,2,18,3,6,3,56,6,5
#q1,q2,q3,a1,a2,a3,a4,a5,a6,a7,a8,a9=0.2,0.2,0.2,4,2,18,3,6,3,56,6,5
#q1,q2,q3,a1,a2,a3,a4,a5,a6,a7,a8,a9=0.1,0.1,0.1,4,2,18,3,6,3,56,30,5
#q1,q2,q3,a1,a2,a3,a4,a5,a6,a7,a8,a9=0.2,0.2,0.2,4,1,18,1,6,2,56,30,5
#q1,q2,q3,a1,a2,a3,a4,a5,a6,a7,a8,a9=0.1,0.2,0.1,4,8,18,1,6,2,56,10,5
#q1,q2,q3,a1,a2,a3,a4,a5,a6,a7,a8,a9=0.05,0.05,0.05,4,2,18,3,6,3,56,10,40

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
