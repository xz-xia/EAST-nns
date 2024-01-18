import os
import pandas as pd
import numpy as np
import shutil
def rename_dir(work_dir,old_ext,new_ext):
# old_ext, new_ext = '.docx', '.txt'
    for filename in os.listdir(work_dir):
        # 获取得到文件后缀
        everypath = os.path.join(work_dir,filename)
        for filename_1 in os.listdir(everypath):
            everypath_1 = os.path.join(everypath,filename_1)
            for everypath_2 in os.listdir(everypath_1):
                split_file = os.path.splitext(everypath_2)
                file_ext = split_file[1]    # 把所有文件属性(.docx/.txt)赋给file_ext

                if old_ext == file_ext:     # 如果文件属性是 .docx 执行
                    newfile = split_file[0] + new_ext  # 修改后的文件完整名称
                    os.rename( # 实现重命名操作
                        os.path.join(everypath_1, everypath_2), # 文件路径不变
                        os.path.join(everypath_1, newfile)) # 文件后缀变为 [new_ext]值
#                print("完成重命名")
#    print(os.listdir(work_dir)) # 打印修改后文件信息
    return