import numpy as np
def X_point(psirz):
    #===============输入准备结束==================
    R=np.linspace(1.2,2.8,129)
    dr = R[1]-R[0]
    Z=np.linspace(-1.4,1.4,129)
    dz = Z[1]-Z[0]
    m=psirz.shape[0]
    #print("psirz.shape[0]:",m)
    n=psirz.shape[1]
    #print("psirz.shape[1]:",n)
    Bp=np.zeros([m,n])
#==========================计算部分开始=========================================
# 1计算得到Br,Bz,Bp
    #转置后psirz的行表示不同的Z，列表示不同的R，第三维为时间，与极向截面坐标一致。
    #[dpsidR(:,:),dpsidZ(:,:)] =np.gradient(psirz,R,Z);#这是我原本的计算公式，发现与gfile中的结果对应不上，正好正负相反。
    dpsidZ, dpsidR = np.gradient(psirz, -dz, -dr)
    #[dpsidR1(:,:),dpsidZ1(:,:)] = gradient(psirz,R,Z);%
    #以上操作之后就与Gfile中的Br和Bz对应上了；具体原因应该是梯度前面加一个负号，意味着梯度向相反的方向前进；
    #RR=ones(size(R))*R'
    tempRR=np.repeat(R, len(Z), axis=0)
    RRtemp=tempRR.reshape(len(Z),len(R))
    RR=RRtemp.T#对RR进行转置成行等差，列相同
    Bz=dpsidR/RR#tokamka pages:108
    #
    Br=-1*dpsidZ/RR#tokamka pages:108
    Bp=np.sqrt(np.square(Br)+np.square(Bz))#平方后开方就是Bp
    #=============================找到上X点==================================
    Iup=np.array(np.where(Z>0.4))
    Zup=Z[Iup]
    #Bp=np.array(Bp)
    BpupX=Bp[Iup[0],:]#上X点矩阵
    #plt.contour(R,Zup[0],BpupX,50)
    #===============找到BpupX 矩阵最小值位置=================================
    atuple=np.where(BpupX==BpupX.min())
    rowup=atuple[0][0]#BpupX最小值行的索引值
    columnup=atuple[1][0]#BpupX最小值列的索引值
    minBpupX=BpupX.min()#输出BpupX的最小值
    zx0up=Zup[0][rowup]#上X点的Z坐标
    rx0up=R[columnup]#上X点的R坐标
    btuple=np.where(Bp==minBpupX) #找到上X点在R，Z矩阵中的位置
    indrow=btuple[0][0] #找到上X点在R，Z矩阵中的位置行索引值
    indcolumn=btuple[1][0]#找到上X点在R，Z矩阵中的位置列索引值
    #R[indcolumn],Z[indrow]
    #=============求grad Br============
    dBrdz,dBrdr = np.gradient(Br, dz, dr)#分别求r,z梯度
    dBzdz,dBzdr=np.gradient(Bz, dz, dr)
    A=dBrdr[indrow,indcolumn]*dBzdz[indrow,indcolumn]-dBzdr[indrow,indcolumn]*dBrdz[indrow,indcolumn]
    delta_r=(Bz[indrow,indcolumn]*dBrdz[indrow,indcolumn]-Br[indrow,indcolumn]*dBzdz[indrow,indcolumn])/A
    delta_z=(Br[indrow,indcolumn]*dBzdr[indrow,indcolumn]-Bz[indrow,indcolumn]*dBrdr[indrow,indcolumn])/A
    #=============计算上x点的rz位置的误差=======================================
    rxup=rx0up+delta_r#X点的r坐标
    zxup=zx0up+delta_z#X点的z坐标
    return rxup , zxup