import os
#批量删除文件夹函数
def delete_function(file_dir):
    path_1 = os.listdir(file_dir)#数字文件
    e = []
    for path_2 in path_1:
        path_3 = os.path.join(file_dir,path_2,'BT')
        path_4 = os.path.join(file_dir,path_2,'PSIRZ')
        path_3_1 = os.listdir(path_3)
        path_4_1 = os.listdir(path_4)
        a = []#存放bt的title
        b = []#存放psirz的
        c = []#存放两文件相等的情况
        for path_5 in path_3_1:
            a.append(path_5)
        for path_6 in path_4_1:
            b.append(path_6)
        c = list(set(a)-set(b))
        e.append(c)
    i = 0#用于遍历获取的多余项列表
    n = 0#用于计数删除的文件数
    for path_2 in path_1:
        path_3 = os.path.join(file_dir,path_2)
        path_4 = os.listdir(path_3)
        for path_5 in path_4:
            if os.path.splitext(path_5)[-1] != '.npy':
                path_6 = os.path.join(path_3,path_5)
                path_7 = os.listdir(path_6)
                for path_8 in path_7:
                    if path_8 in e[i]:
                        os.remove(os.path.join(path_6,path_8))
                        print("已经删除：",os.path.join(path_6,path_8))
                        n = n+1
        i = i+1
    print("已经删除文件个数：",n/4)
    return