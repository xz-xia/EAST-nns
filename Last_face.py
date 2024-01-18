import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
def Last_face(psirz,rxup,zxup):
#================上X点位置与误差计算完毕==================
    R=np.linspace(1.2,2.8,129)
    dr = R[1]-R[0]
    Z=np.linspace(-1.4,1.4,129)
    dz = Z[1]-Z[0]
    m=psirz.shape[0]
    #print("psirz.shape[0]:",m)
    n=psirz.shape[1]
    #print("psirz.shape[1]:",n)
    Bp=np.zeros([m,n])
# 2找到最后闭合磁面位置：找到与X点相同的磁通值对应的位置；
    #首先找到X点对应的极向磁通值；
    Rinterp=np.linspace(min(R),max(R),2**11+1)
    Zinterp=np.linspace(min(Z),max(Z),2**11+1)
    Rtest=np.repeat(Rinterp, 2**11+1, axis=0)
    Rtest1=Rtest.reshape(2**11+1,2**11+1)
    RRinterp=Rtest1.T#R坐标需要每一列都一样
    Ztest=np.repeat(Zinterp, 2**11+1, axis=0)
    ZZinterp=Ztest.reshape(2**11+1,2**11+1)#R坐标需要每一列都一样
    f1=interpolate.interp2d(R,Z,psirz,kind='cubic')
    psiinterp = f1(Rinterp, Zinterp)#
    f2=interpolate.interp2d(Rinterp,Zinterp,psiinterp,'cubic')
    psi_Xpoint=f2(rxup,zxup)#X点处的极向磁通值
    psi_Xpoint=psi_Xpoint[0]
# 3获得最后闭合磁面的位置；
# 等值线绘图及提取数据示例
    cs = []
    for i in range(0,100):
        dpsi=0
        dpsi=dpsi+1e-6*i
        cs.append(plt.contour(RRinterp, ZZinterp, psiinterp,levels=[psi_Xpoint+dpsi-30*1e-6]))
        plt.close()
        if len(cs[i].collections[0].get_paths())==2:
            if i < 4:
                return 0
            p = cs[i-4].collections[0].get_paths()[2]#取最后闭合磁面那条线路径
            v = p.vertices#获取该条路径上坐标
            r_last = v[:,0]#获取最后闭合磁面的r坐标
            z_last = v[:,1]#获取最后闭合磁面的z坐标
            print("index:",i)
            print("cs[i]:",len(cs[i].collections[0].get_paths()))
            print("cs[i-4]:",len(cs[i-4].collections[0].get_paths()))
            return v
        print(i)