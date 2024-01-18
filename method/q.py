import os
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import griddata
from scipy.interpolate import splprep, splev
import sys
import matplotlib.path as mpath
from scipy.integrate import dblquad
import time
from joblib import Parallel, delayed
import X_point
import Last_face
def get_contour_verts(cn):
    contours = []
    idx = 0
    # for each contour line
    #print(cn.levels)
    for cc,vl in zip(cn.collections,cn.levels):
        # for each separate section of the contour line
        for pp in cc.get_paths():
            paths = {}
            paths["id"]=idx
            paths["type"]=0
            paths["value"]=float(vl) # vl 是属性值
            xy = []
            # for each segment of that section
            for i, vv in enumerate(pp.vertices):
                if i%1 == 0:#整除最后闭合磁面的点减少原来的8倍
                    xy.append([float(vv[0]),float(vv[1])]) #vv[0] 是等值线上一个点的坐标，是 1 个 形如 array[12.0,13.5] 的 ndarray。
            paths["coords"]=xy
            contours.append(paths)
            idx +=1
    return contours

def contains_point(path, point):
    return path.contains_point(point)

def func(y, x):
    return x*y
def Q(psirz,Bt):
    tic= time.perf_counter()
    rxup,zxup = X_point(psirz)
    rlcfs,zlcfs = Last_face(psirz,rxup,zxup)
    BT=Bt
    R = np.linspace(1.2, 2.8, 129)
    Z = np.linspace(-1.4, 1.4, 129)
    RR = np.outer(np.ones_like(Z), R)
    ZZ = np.outer(Z, np.ones_like(R))
    psirztemp = psirz[45:87, 33:99]
    RRtemp = RR[45:87, 33:99]
    valueo, indo = np.min(psirztemp), np.argmin(psirztemp)
    a, b = np.where(psirztemp == valueo)
    Rmaxis = RRtemp[a[0], b[0]]
    It = np.abs(BT)*Rmaxis/(4.16e-4)
    Btr = It*4.16e-4/R
    Btr1 = np.tile(Btr, (len(R), 1))


#=====================找到对应LCFS 处的极向磁通值==============================
    f2=interpolate.interp2d(R,Z,psirz,kind='cubic')
    result = []
    for r, z in zip(rlcfs, zlcfs):
        lcfspsi = f2(r, z)
        result.append(lcfspsi)
    lcfspsirz = np.array(result)[:,0]
    psiXpoint=lcfspsirz[0]

# ==================================2======================================
# magnetic flux at the magnetic axis
    psipmag = valueo

# magnetic flux at the outermost closed magnetic surface
    psiX = psiXpoint

# magnetic flux between the outermost closed magnetic surface and the magnetic axis
    psipbm = psiX - psipmag

# determine the number of magnetic surfaces
    nf = 33

# variation of poloidal magnetic flux
# find nf positions between the magnetic axis and the outermost closed magnetic surface
# first, divide the poloidal magnetic flux into nf parts
    psicimian = psipbm * np.linspace(0, 1, nf)

    cimian20ge = {}
    s1={}
    for j in range(1, nf):

        if j == nf:
            psipoint = psipmag + psicimian[j] + 1e-7
        else:
            psipoint = psipmag + psicimian[j]

        c = plt.contour(R, Z, psirz, levels=[psipoint])
        s=get_contour_verts(c)
    #ss.append(s)
        for jj in range(len(s)):

            clf = s[jj]
            rlcfs=np.array(list(s[jj].values())[3]) [:,0]
            zlcfs=np.array(list(s[jj].values())[3]) [:,1]
            if all(x > 0 for x in zlcfs):
                print(0)
            elif all(x <0 for x in zlcfs):
                print(0)
            else:
                print(1)
                s1 = clf

        tempx = np.zeros((len(np.array(list(s1['coords']))[:,0]), 2))
        tempx[:, 0] = np.array(list(s1['coords']))[:,0]
        tempx[:, 1] = np.array(list(s1['coords']))[:,1]
     # 定义一个空字典作为初始值
        mystr = 'ind' + str(j)
        tempj=tempx
        cimian20ge[mystr] = tempj
        del s,s1, tempx

#=============找到这nf个磁面对应的Bphi的值的坐标不止一个很多个
    Btcimian={}
    BtcimianRinterp={}
    BtcimianZinterp={}
    Btcimianinterp={}
    for kk in range(1, nf):
        mystr = 'ind' + str(kk)
        temp = cimian20ge[mystr]
        tempR = temp[:,0]
        tempZ = temp[:,1]
        f2=interpolate.interp2d(R,Z,Btr1,kind='cubic')
        result = []
        for r, z in zip(tempR, tempZ):
            temptemprz = f2(r, z)
            result.append(temptemprz)
        tempBt = np.array(result)[:,0]
        Btcimian[mystr] = tempBt

    #将这个磁面上的点插值成1000个

        datatemp=np.array([tempR,tempZ])
        tck, u = splprep(datatemp, u=None, s=0.0, per=1)
        u_new = np.linspace(u.min(), u.max(), 1000)
        tempx, tempy = splev(u_new, tck, der=0)
        result = []
        for r, z in zip(tempx, tempy):
            temptemprz = f2(r, z)
            result.append(temptemprz)
        tempinterpBt = np.array(result)[:,0]

        BtcimianRinterp[mystr]=tempx
        BtcimianZinterp[mystr]=tempy
        Btcimianinterp[mystr]=tempinterpBt
        del tempR, tempZ, tempBt, tempx, tempy, temp

#==============%计算曲线的面积并计算环向磁通====================================
    Ngrid=200
    As = np.zeros((nf, 1))
    Asgrid = np.zeros((nf, 1))
    phi = np.zeros((nf, 1))
    psi = np.zeros((nf, 1))
    Rcimian = np.zeros((nf,))
    BtrtempRZ=[]
    psitempRZ=[]
    psitempRZtest=[]
    BtrtempRZtest =[]

    for nn in range(1, nf):
        #print(nn)
        dphi = np.empty((Ngrid, Ngrid))
        dpsi = np.empty((Ngrid, Ngrid))
        BtrtempRZtest = np.empty((Ngrid, Ngrid))
        mystr_ = 'ind' + str(nn)
        tempBt=Btcimianinterp[mystr_]
        tempR=BtcimianRinterp[mystr_]
        tempZ=BtcimianZinterp[mystr_]
    # 计算环向磁通
        tempRR = np.linspace(min(tempR), max(tempR), Ngrid)
        tempZZ = np.linspace(min(tempZ), max(tempZ), Ngrid)
        tempZZZZ, tempRRRR= np.meshgrid(tempZZ, tempRR)
        tempRRRR1=tempRRRR.T
        tempZZZZ1=tempZZZZ


    # 将原始坐标和数据打包成一个(n_samples, 3)的数组
        points = np.column_stack([RR.flatten(), ZZ.flatten(), Btr1.flatten()])
        points2=np.column_stack([RR.flatten(), ZZ.flatten(), psirz.flatten()])
    # 将新坐标打包成一个(n_samples, 2)的数组
        xi=[]
        xi = np.column_stack([tempRRRR1.flatten(), tempZZZZ1.flatten()])
    # 对数据进行插值
        Btr1_interp = griddata(points[:, :2], points[:, 2], xi, method='nearest')#nearest
        BtrtempRZ = np.reshape(Btr1_interp, (Ngrid, Ngrid))
        psirz1_interp = griddata(points2[:, :2], points2[:, 2], xi, method='nearest')
        psitempRZ = np.reshape(psirz1_interp, (Ngrid, Ngrid))
        psitempRZtest = psitempRZ.copy()
        BtrtempRZtest = BtrtempRZ.copy()
        verts = np.array([tempR, tempZ])
        vertsT = verts.T
        path = mpath.Path(vertsT)

        num_cores = 3
        # 并行执行for循环
        results = Parallel(n_jobs=num_cores)(
            delayed(contains_point)(path, [tempRRRR1[i, j], tempZZZZ1[j, i]])
            for i in range(len(tempRRRR1))
            for j in range(len(tempRRRR1))
        )

    # 更新数组
        results = np.array(results).reshape((200, 200))
        psitempRZtest[~results] = 0
        BtrtempRZtest[~results] = 0


        xmin, xmax = np.min(tempR), np.max(tempR)
        ymin, ymax = np.min(tempZ), np.max(tempZ)
        area, err = dblquad(func, ymin, ymax, lambda x: xmin, lambda x: xmax)
        #print("Closed curve area =", area)
        As[nn, 0] = area
 # plt.plot(tempR, tempZ, 'r', linewidth=2)
        Asgrid[nn, 0] = (max(tempRRRR1[0, :]) - min(tempRRRR1[0, :])) * (max(tempZZZZ1[0, :]) - min(tempZZZZ1[0, :]))
        ds = Asgrid[nn, 0] / (len(tempRRRR1) * len(tempZZZZ1))
        dphi = np.zeros((len(tempRRRR1), len(tempZZZZ1)))
        dpsi = np.zeros((len(tempRRRR1), len(tempZZZZ1)))
        for m in range(len(tempRRRR1)):
            for n in range(len(tempZZZZ1)):
               dphi[m, n] = BtrtempRZtest[m, n] * ds  # dphi
               dpsi[m, n] = psitempRZtest[m, n]

        phi[nn, 0] = np.sum(np.sum(dphi))
        psirztest = psirz.copy()
        f5= interpolate.interp2d(R, Z,psirztest, kind='cubic')
        result = []
        for r, z in zip(tempR, tempZ):
            temptemprz = f5(r, z)
            result.append(temptemprz)
        psitemp= np.average(np.array(result)[:,0]) # 计算极向磁通

        psi[nn, 0] = psitemp
        Rcimiantemp = cimian20ge[mystr_][:,0]
        Rcimian[nn] = np.max(Rcimiantemp)
        plt.close('all')

    dphi_f = np.gradient(phi[:,0])
    dpsi_f=np.gradient(psi[:,0]);
    qcal=abs(dphi_f /dpsi_f/2/math.pi);
    toc= time.perf_counter()
    print(f"Elapsed time: {toc - tic:0.6f} seconds")
    #plt.plot(Rcimian[2:],qcal[2:],'b-o')
    return np.hstack((Rcimian[2:].reshape(len(Rcimian[2:]),1),qcal[2:].reshape(len(qcal[2:]),1)))