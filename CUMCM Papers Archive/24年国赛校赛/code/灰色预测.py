import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc("font",family='FangSong')

def GM11(X0, t, k):
    # 第一步，一次累加生成序列 X(1)
    print("\n为弱化原始时间序列的随机性，采用累加的方式处理数据\n")
    X1 = np.cumsum(X0)
    print("原始序列X(0)为:", X0)
    print("累加序列X(1)为:", X1, '\n')

    # 第二步，开始构造矩阵 B 和数据向量 Yn
    b = -(X1[:-1] + X1[1:]) / 2
    D = np.ones(t-1)
    B = np.column_stack((b, D))
    YN = X0[1:]

    # 第三步，开始计算待估参数向量 alpha2，利用最小二乘法求解，以便获得发展灰数 a 和内生控制灰数 u
    alpha = np.linalg.inv(B.T @ B) @ B.T @ YN
    a = alpha[0]
    u = alpha[1]
    print("GM(1,1)参数估计值：发展系数a =", a, "灰色作用量u =", u, '\n')

    # 第四步，开始计算 X(0) & X(1) 的预测序列
    Xhat1 = np.zeros(t)
    Xhat1[0] = X1[0]
    for i in range(1, t):
        Xhat1[i] = (X0[0] - u / a) * np.exp(-a * i) + u / a

    Xhat0 = np.zeros(t)
    Xhat0[0] = Xhat1[0]
    for o in range(1, t):
        Xhat0[o] = Xhat1[o] - Xhat1[o-1]

    print("X(1)的模拟值:", Xhat1)
    print("X(0)的模拟值:", Xhat0, '\n')

    # 第五步，开始进行残差检验
    print("开始进行残差检验：", '\n')
    e = X0 - Xhat0
    r_e = np.abs(e) / X0
    e_average = np.mean(r_e)

    print("绝对残差：", '\n', "         ", np.round(np.abs(e), 6))
    print("相对残差:", '\n', "         ", np.round(r_e, 8), '\n')
    print("残差平方和 =", np.sum(e**2), '\n')
    print("平均相对误差 =", e_average * 100, "%", '\n')
    print("相对精度 =", (1 - e_average) * 100, "%", '\n')

    if (1 - e_average) * 100 > 90:
        print("模型精确度较高，通过残差检验", '\n')
    else:
        print("模型精确度未高于90%，残差检验未通过", '\n')
        return

    # 第六步，开始进行关联度检验
    print("开始进行关联度检验：", '\n')
    pho = 0.5
    eta = (np.min(np.abs(e)) + pho * np.max(np.abs(e))) / (np.abs(e) + pho * np.max(np.abs(e)))
    r = np.round(np.mean(eta), 2)

    print("关联度为:r =", r, '\n')
    if r > 0.55:
        print("满足 pho=0.5 时的检验准则 r>0.55，通过关联度检验", '\n')
    else:
        print("未通过关联度检验", '\n')
        return

    # 第七步，开始进行后验差比值检验
    print("开始进行后验差检验：", '\n')
    eavge = np.mean(np.abs(e))
    se = np.std(np.abs(e), ddof=1)
    X0avge = np.mean(X0)
    sx = np.std(X0, ddof=1)
    c_value = se / sx
    S0 = np.sum((np.abs(e) - eavge) < 0.6745 * sx) / len(e)

    print("原始序列X0的标准差=", sx, '\n')
    print("绝对误差的标准差=", se, '\n')
    print("C值=", c_value, '\n')
    print("小误差概率:P值=", S0, '\n')

    if c_value < 0.35 and S0 > 0.95:
        print("C<0.35, P>0.95,GM(1,1)预测精度等级为：好，通过后验差检验", '\n')
    elif c_value < 0.5 and S0 > 0.80:
        print("C值属于[0.35,0.5), P>0.80,GM(1,1)模型预测精度等级为：合格，通过后验差检验", '\n')
    elif c_value < 0.65 and S0 > 0.70:
        print("C值属于[0.5,0.65), P>0.70,GM(1,1)模型预测精度等级为：勉强合格，通过后验差检验", '\n')
    else:
        print("C值>=0.65, GM(1,1)模型预测精度等级为：不合格，未通过后验差检验", '\n')
        return

    # 第八步，画出输入预测序列Xhat0与原始序列X0的比较图像
    # 更改横坐标表示的数据，为2017-2021年而不是0-4，且显示的是整数
    plt.figure(figsize=(10, 6))
    x = np.arange(2017, 2022, 1)
    plt.plot(x, Xhat0, color='#B5A1E3', marker='o', linestyle='-', label='预测')
    plt.plot(x, X0, color='#F0C2A2', marker='x', linestyle='-', label='原始')
    plt.xlabel('年份', fontsize=13)
    plt.ylabel('互联网用户数/千', fontsize=13)
    plt.xticks(x, fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(loc='best', title='原始序列与预测序列的对比', fontsize=13)
    plt.show()

    # 第九步，求出第k期的预测值 Xhat0[k]
    Xhat1_k = (X0[0] - u / a) * np.exp(-a * (k-1)) + u / a
    Xhat1_k_minus_1 = (X0[0] - u / a) * np.exp(-a * (k-2)) + u / a
    Xhat0_k = Xhat1_k - Xhat1_k_minus_1

    print("第", k, "期的预测值为:", Xhat0_k)

x = np.array([319, 387, 450, 493, 546])

x = np.array([319, 387, 450, 493, 546])
GM11(x, len(x), 7)