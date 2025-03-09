import random
number=0
lst11=[]#存储决策
lst12=[]#存储利润
lst141,lst151=[],[]
for i in range(10000):
    lst131=[]
    
    # try:
    def three_1(n1,q1,G1,H1,I1,q4=0.1):
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

    def two_1(n1,q1,G1,H1,I1,q4=0.1):
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
    c21,c31,c22_1,c41_1,N,q,A1,A2,A3,A4,A5=three_1(M1,V1,G1,H1,I1,q4=0.1)
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
        c21,c31,c22_1,c41_1,N,q,AA1,AA2,AA3,AA4,AA5=three_1(N,q,G1,H1,I1,q4=0.1)
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
    c21,c31,c22_1,c41_1,N,q,B1,B2,B3,B4,B5=three_1(M2,V2,G2,H2,I2,q4=0.1)
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
        c21,c31,c22_1,c41_1,N,q,AA1,AA2,AA3,AA4,AA5=three_1(N,q,G2,H2,I2,q4=0.1)
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
    c21,c31,c22_1,c41_1,N,q,O1,O2,O3,O4=two_1(M3,V3,G3,H3,I3,q4=0.1)
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
        c21,c31,c22_1,c41_1,N,q,OO1,OO2,OO3,OO4=two_1(N,q,G3,H3,I3,q4=0.1)
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
    q2=0.1
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
            c21,c31,c22_1,c41_1,N,q,A1_,A2,A3,A4,A5=three_1(M1,V1,G1,H1,I1,q4=0.1)
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
                c21,c31,c22_1,c41_1,N,q,AA1,AA2,AA3,AA4,AA5=three_1(N,q,G1,H1,I1,q4=0.1)
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
            c21,c31,c22_1,c41_1,N,q,B1_,B2,B3,B4,B5=three_1(M2,V2,G2,H2,I2,q4=0.1)
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
                c21,c31,c22_1,c41_1,N,q,AA1,AA2,AA3,AA4,AA5=three_1(N,q,G2,H2,I2,q4=0.1)
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
            c21,c31,c22_1,c41_1,N,q,O1_,O2,O3,O4=two_1(M3,V3,G3,H3,I3,q4=0.1)
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
                c21,c31,c22_1,c41_1,N,q,AA1,AA2,AA3,AA4=two_1(N,q,G3,H3,I3,q4=0.1)
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
        
        if S>0 and pi>0 and W<12000:
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
#     m=m+1

# lst11.append(lst10)
# lst12.append(sum(lst4)) 
# lst4=[]
# except ZeroDivisionError:
#     continue
#print(lst12)
# print(len(lst12))
# test1=0
# test2=0
# for i in range(len(lst12)):
# if lst12[i]>test1 and lst12[i]<20000:
#     test1=lst12[i]
#     test2=i
# print(lst11[test2])
# print(test1)


