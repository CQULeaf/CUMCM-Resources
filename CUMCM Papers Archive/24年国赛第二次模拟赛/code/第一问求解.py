#第一问代码
import math
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lstsq

#导入数据
data=pd.read_excel("坐标.xlsx")
X=data['x坐标 (m)']
Y=data['y坐标 (m)']

#入射光线的单位向量
def cal(a,b,c):#纬度，当地时间，距春分时间
    Sunang=math.pi/12*b #太阳时角

    Sundec=math.asin(math.sin(2*math.pi*c/365)*math.sin(2*math.pi/360*23.45))#太阳赤纬角

    Sunalt=math.asin(math.cos(Sundec)*math.cos(a)*math.cos(Sunang)+math.sin(Sundec)*math.sin(a))#太阳高度角

    test=(math.sin(Sundec)-math.sin(Sunalt)*math.sin(a))/(math.cos(Sunalt)*math.cos(a))
    
    if abs(test)>1:
        test=test/abs(test)
    Sunloc=math.acos(test)#太阳方位角

    if b<0:
        Sunloc=Sunloc
    else:
        Sunloc=-Sunloc
    
    x=math.cos(Sunalt)*math.cos(Sunloc)
    y=math.cos(Sunalt)*math.sin(Sunloc)
    z=math.sin(Sunalt)

    return x,y,z
a=math.radians(39.4)
lstb=[-3,-1.5,0,1.5,3]
lstc=[306,337,0,31,61,92,122,153,184,214,245,275]
lstsum=[]
lstsubsum=[]
for i in range(12):
    for j in range(5):
        x,y,z=cal(a,lstb[j],lstc[i])
        lstsubsum.append(x)
        lstsubsum.append(y)
        lstsubsum.append(z)
        lstsum.append(lstsubsum)
        lstsubsum=[]

#计算余弦效率
mirpos=[]
lst=[]

#定义求镜器法向量的函数：
def mirmacvec(x):
    mirpos=[]
    sum=math.sqrt(x[0]**2+x[1]**2+76**2)
    mirpos.append(-x[0]/sum)
    mirpos.append(-x[1]/sum)
    mirpos.append(76/sum)
    return mirpos
    
#计算镜器法向量
for i in range(len(X)):
    sum=math.sqrt(X[i]**2+Y[i]**2+76**2)
    lst.append(-X[i]/sum)
    lst.append(-Y[i]/sum)
    lst.append(76/sum)
    mirpos.append(lst)
    lst=[]
#print(mirpos)

#计算镜面法向量
def Mirvec(lst1):
    lst2=[]
    for i in range(60):
        a=lst1[0]*lstsum[i][0]+lst1[1]*lstsum[i][1]+lst1[2]*lstsum[i][2]
        if a>0:
           lst1[0]=-lst1[0]
           lst1[1]=-lst1[1]
           lst1[2]=-lst1[2]
        sum=math.sqrt((lst1[0]-lstsum[i][0])**2+(lst1[1]-lstsum[i][1])**2+(lst1[2]-lstsum[i][2])**2)
        x=(lst1[0]-lstsum[i][0])/sum
        y=(lst1[1]-lstsum[i][1])/sum
        z=(lst1[2]-lstsum[i][2])/sum
    return x,y,z

#计算镜面法向量和余弦效率
def Norvec(lst1):
    lst2=[]
    for i in range(60):
        a=lst1[0]*lstsum[i][0]+lst1[1]*lstsum[i][1]+lst1[2]*lstsum[i][2]
        if a>0:
           lst1[0]=-lst1[0]
           lst1[1]=-lst1[1]
           lst1[2]=-lst1[2]
        sum=math.sqrt((lst1[0]-lstsum[i][0])**2+(lst1[1]-lstsum[i][1])**2+(lst1[2]-lstsum[i][2])**2)
        x=(lst1[0]-lstsum[i][0])/sum
        y=(lst1[1]-lstsum[i][1])/sum
        z=(lst1[2]-lstsum[i][2])/sum
        cof=abs(x*lstsum[i][0]+y*lstsum[i][1]+z*lstsum[i][2])
        lst2.append(cof)
        averages = []
    for i in range(0, 60, 5):
        chunk = lst2[i:i + 5]
        sum=chunk[0]+chunk[1]+chunk[2]+chunk[3]+chunk[4]
        sum=sum/5
        averages.append(sum)
    return averages

#计算余弦效率
def cos(lst1):
    lst2=[]
    for i in range(60):
        a=lst1[0]*lstsum[i][0]+lst1[1]*lstsum[i][1]+lst1[2]*lstsum[i][2]
        if a>0:
           lst1[0]=-lst1[0]
           lst1[1]=-lst1[1]
           lst1[2]=-lst1[2]
        sum=math.sqrt((lst1[0]-lstsum[i][0])**2+(lst1[1]-lstsum[i][1])**2+(lst1[2]-lstsum[i][2])**2)
        x=(lst1[0]-lstsum[i][0])/sum
        y=(lst1[1]-lstsum[i][1])/sum
        z=(lst1[2]-lstsum[i][2])/sum
        cof=abs(x*lstsum[i][0]+y*lstsum[i][1]+z*lstsum[i][2])
        lst2.append(cof)
    return lst2
lst23=[]

#计算60个时间点的平均余弦效率
for i in range(len(X)):
    lst23=lst23+cos(mirpos[i])
for i in range(60):
    lst23[i]=lst23[i]/len(X)    #60个时间点的平均余弦效率  

sumsum=[]
for i in range(len(mirpos)):
    sumsum.append(Norvec(mirpos[i]))

#得到春分和夏至的结果目的是可视化
spring,summer=[],[]
for i in range(len(mirpos)):
    spring.append(sumsum[i][2])
    summer.append(sumsum[i][5])
positions=list(zip(X,Y))
plt.scatter([x for x, _ in positions], [y for _, y in positions], c=spring, cmap='viridis')
plt.colorbar(label='Value')
#plt.show()

#计算平均效率
avgcof=[]
for i in range(12):
    sum2=0
    for j in range(len(mirpos)):
       sum2=sum2+sumsum[j][i]
    avg=sum2/len(mirpos)
    avgcof.append(avg)
#print(avgcof)


lst2=[]
positions3=[]
#计算阴影遮挡效率
for i in range(len(X)):
    lst2.append(4)
positions2=list(zip(X,Y,lst2))
for x in positions2:
    x=list(x)
    positions3.append(x)#储存了所用定日镜的中心位置


#求镜子俯仰角和方位角
def calpos(x):
    a,b,c=Mirvec(x)
    theta1=math.acos(c)#俯仰角
    theta2=math.atan(a/b)#方位角
    return theta1,theta2

#求镜子25个格子各中心点的坐标,先旋转再平移
def midpos(x):
    lst1=[]
    lst4=[]
    theta1,theta2=calpos(x)
    for i in range(5):
        a=-2.4+1.2*i
        for j in range(5):
            lst2=[]
            b=2.4-1.2*j
            lst2.append(a)
            lst2.append(b)
            lst2.append(0)
            lst1.append(lst2)

    matrix1=np.array([[math.cos(theta1),0,math.sin(theta1)],[0,1,0],[-math.sin(theta1),0,math.cos(theta1)]])
    matrix2=np.array([[math.cos(theta2),-math.sin(theta2),0],[math.sin(theta2),math.cos(theta2),0],[0,0,1]])
    matrix3=matrix1@matrix2
    # print(matrix3)
    # print(matrix3.shape)
    # print(np.array(lst1[0]))
    # print(np.array(lst1[0]).reshape(-1,1))
    # vec=np.array(lst1[0]).reshape(-1,1)
    #print(matrix3.dot(vec))

    #计算四个顶点的坐标
    lst5=[[-3,3,0],[3,3,0],[-3,-3,0],[3,-3,0]]
    lst1=lst1+lst5
    for i in range(len(lst1)):
        lst3=[]
        vec=np.array(lst1[i]).reshape(-1,1)
        w=matrix3.dot(vec)
        l,m,n=w[0][0],w[1][0],w[2][0]
        lst3.append(l)
        lst3.append(m)
        lst3.append(n)
        lst4.append(lst3)
    for i in range(len(lst1)):
        lst4[i][0]=lst4[i][0]+x[0]
        lst4[i][1]=lst4[i][1]+x[1]
        lst4[i][2]=lst4[i][2]+x[2]
    return lst4

def is_point_in_triangle(p, triangle):
    """
    判断点 p 是否在三角形 triangle 内部
    :param p: 测试点坐标 (x, y)
    :param triangle: 三角形顶点坐标 [(x1, y1), (x2, y2), (x3, y3)]
    :return: True 如果点 p 在三角形内部，否则 False
    """
    AB = np.array(triangle[1]) - np.array(triangle[0])
    BC = np.array(triangle[2]) - np.array(triangle[1])
    CA = np.array(triangle[0]) - np.array(triangle[2])

    AP = np.array(p) - np.array(triangle[0])
    BP = np.array(p) - np.array(triangle[1])
    CP = np.array(p) - np.array(triangle[2])

    detABAP = np.cross(AB, AP)
    detBCBP = np.cross(BC, BP)
    detCAPC = np.cross(CA, CP)

    if np.dot(detABAP,detBCBP)>0 and np.dot(detABAP,detCAPC)>0 and np.dot(detBCBP,detCAPC )>0:
       return True
    else:
        return False
# print(tuple(np.array(lstsum[0])))
# print(np.array(Mirvec(positions3[1])))   
# print(np.array(midpos(positions3[0])[0]))  
# print(np.array(positions3[0]))  

def line_plane_intersection(n, p0, d, q0):
    # n 是平面的法向量 (a, b, c)
    # p0 是平面上的一点 (x0, y0, z0)
    # d 是直线的方向向量 (d1, d2, d3)
    # q0 是直线上的一点 (u0, v0, w0)
    
    a, b, c = n
    x0, y0, z0 = p0
    u0, v0, w0 = q0
    d1, d2, d3 = d
    
    # 计算平面方程的常数项
    C = a * x0 + b * y0 + c * z0
    
    # 计算直线参数方程代入平面方程后的系数
    coeff_t = a * d1 + b * d2 + c * d3
    const = C - (a * u0 + b * v0 + c * w0)
    
    # 如果系数为0，直线可能与平面平行或在平面内
    if coeff_t == 0:
        if const == 0:
            print("直线在平面内，有无穷多交点")
            return None
        else:
            print("直线与平面平行，无交点")
            return None
    
    # 计算参数t
    t = -const / coeff_t
    
    # 计算交点坐标
    intersection_point = (u0 + d1 * t, v0 + d2 * t, w0 + d3 * t)
    return intersection_point
def shadow(x):
    print(1)
    lst6=[]
    n=0
    lst5=midpos(x)
    lst7=lst5  #统计在某一时刻阴影下的有效点
    lst13=[]  #60个时间点阴影下的有效点
    for i in range(60):
        for j in range(25):
            # 直线上的一点
            point_on_line = np.array(lst5[j])
            # 直线的方向向量
            direction_vector = np.array(lstsum[i])
            for k in range(len(X)):
                if positions3[k]==x:
                    continue
                else:
                    normal_vector=np.array(Mirvec(positions3[k]))
                    point_on_plane = np.array(positions3[k])
                    lst10=midpos(positions3[k])
                    intersection_point=line_plane_intersection(tuple(normal_vector),tuple(point_on_plane),tuple(direction_vector),tuple(point_on_line))
                    triangle1=[tuple(lst10[21]),tuple(lst10[22]),tuple(lst10[23])]
                    triangle2=[tuple(lst10[22]),tuple(lst10[23]),tuple(lst10[24])]
                    if is_point_in_triangle(intersection_point, triangle1) or is_point_in_triangle(intersection_point, triangle2):
                        lst7.remove(lst5[j])
                        n=n+1 
                        break
        lst13.append(lst7)
        lst7=lst5
        lst6.append(n)
        n=0
    return lst6,lst13
     
def block(x):
    lst8,lst9=shadow(x)
    lst12=lst9#表示在阴影和遮挡下都有效的点
    lst14=[]
    m=0
    for j in range(len(lst9)):
        for l in range(len(lst9[j])):#遍历阴影下的有效点
            print(2)
            # 直线上的一点
            point_on_line = np.array(lst9[j][l])
            # 直线的方向向量
            direction_vector = np.array(mirmacvec(x))
            for k in range(len(X)):
                if positions3[k]==x:
                    continue
                else:
                    normal_vector=np.array(Mirvec(positions3[k]))
                    point_on_plane = np.array(positions3[k])
                    lst11=midpos(positions3[k])
                    intersection_point=line_plane_intersection(tuple(normal_vector),tuple(point_on_plane),tuple(direction_vector),tuple(point_on_line))
                    triangle1=[tuple(lst11[21]),tuple(lst11[22]),tuple(lst11[23])]
                    triangle2=[tuple(lst11[22]),tuple(lst11[23]),tuple(lst11[24])]
                    if is_point_in_triangle(intersection_point, triangle1) or is_point_in_triangle(intersection_point, triangle2):
                        m=m+1
                        lst12[j].remove(lst9[j][l])
                        break
    lst14.append(m)
    m=0
    return lst14,lst12,lst8
# lst15_1,lst15_2,lst15_3=block(positions3[0])
# #print(mirmacvec(positions3[0]))
# print(lst15_1+lst15_3)
sample_size=170
random_indices=np.random.choice(len(X),size=sample_size,replace=False)
sample_elements=[positions3[i] for i in random_indices]
print(sample_elements)
lst16,lst17,lst18,lst19,lst20,lst21=[],[],[],[],[],[]
for i in range(170):
    lst16,lst17,lst18=block(sample_elements[i])
    for i in range(len(lst16)):
        lst16[i]=(lst16[i]+lst18[i])/25
    for i in range(len(lst16)):
        lst19.append(1-lst16[i])
    lst20=lst20+lst19
    lst21.append(lst17) #储存所有定日镜60个时间节点有效点
lst20=lst20/170#得到60个时间节点的平均阴影遮挡效率
print(lst20)

#求直线与圆柱面是否有交点
def does_line_intersect_cylinder(line_point, line_direction):
    # 圆柱面的中心、半径和高度范围
    x0, y0, z0 = (0,0,76)
    R =3.5
    z_min, z_max =(72,80) 
    
    # 直线上的点及其方向
    a_x, a_y, a_z = line_point
    b_x, b_y, b_z = line_direction
    
    # 二次方程的系数
    A = b_x**2 + b_y**2
    B = 2 * (b_x*(a_x-x0) + b_y*(a_y-y0))
    C = (a_x-x0)**2 + (a_y-y0)**2 - R**2
    
    # 解二次方程
    delta = B**2 - 4*A*C
    
    # 检查是否存在实数解
    if delta >= 0:
        # 至少存在一个实数解
        # 检查这些解对应的 z 坐标是否在圆柱的高度范围内
        t1 = (-B + np.sqrt(delta)) / (2*A)
        t2 = (-B - np.sqrt(delta)) / (2*A)
        
        z1 = a_z + t1*b_z
        z2 = a_z + t2*b_z
        
        # 检查任一 z 坐标是否在高度范围内
        if (z_min <= z1 <= z_max) or (z_min <= z2 <= z_max):
            return True  # 相交
    
    return False  # 不相交

#计算截断效率
q,p=0,0
lst22=[]
for i in range(60):
    for j in range(len(X)):
        for k in range(len(lst21[j][i])):
            if does_line_intersect_cylinder(tuple(lst21[j][i][k]),tuple(mirpos[j])):
                q=q+1
                break
        p=p+(len(lst21[j][i])-q)/len((lst21[j][i]))
        q=0
    lst22.append(p/len(X)) #60个时间节点的截断效率
print(lst22)

#计算大气透射率
def aircof(x):
    dis=x[0]**2+x[1]**2+76**2
    return 0.99321-0.0001176*dis+1.97e-8*dis**2
lst24,sum3=[],0
for i in range(60):
    for j in range(len(X)):
        sum3=sum3+aircof(positions3[j])
    sum3/len(X)
    lst24.append(sum3)
    sum3=0

#计算光学效率   
suncof=[]    
for i in range(60):
    suncof.append(lst20[i]*lst23[i]*lst24[i]*lst22[i]*0.92)
print(suncof)

#计算太阳高度角
def sunpos(a,b,c):
    Sunang=math.pi/12*b #太阳时角

    Sundec=math.asin(math.sin(2*math.pi*c/365)*math.sin(2*math.pi/360*23.45))#太阳赤纬角

    Sunalt=(math.cos(Sundec)*math.cos(a)*math.cos(Sunang)+math.sin(Sundec)*math.sin(a))#太阳高度角

    return Sunalt
lst25=[]#储存60个时间的太阳高度角
for i in range(12):
    for j in range(5):
        Sunalt=sunpos(a,lstb[j],lstc[i])
        lst25.apppend(Sunalt)
#计算定日镜场的平均输出热功率
lst26=[]#储存60个时间DNI
a1=0.4237-0.00821*3**2
b1=0.5055+0.00595*3.5**2
c1=0.2711+0.01858*0.5**2
for i in range(60):
    lst26.append(1.366*(a1+b1*math.exp(-c1/lst25[i]))) 
sum4=0
for i in range(60):
    suncof[i]=suncof[i]*36
    sum4=sum4+suncof[i]
lst27=[]#60个时间点的光学效率
for i in range(60):
    lst27.append(lst26[i]*sum4)