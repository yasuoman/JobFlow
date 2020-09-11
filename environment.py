# project : Job_Flow
# file   : environment.py
# author:yasuoman
# datetime:2020/7/22 11:03
# software: PyCharm

"""
description：Information about the environment
说明：关于产生加工工件的初始时间,机器和工件个数
"""
import numpy as np

def create_time_tables():
    # np.random.seed(3)  #产生的是伪随机
    #机器数量
    m =2
    #工件数量
    n =10
    time_tables = np.random.randint(low=1,high=100,size=(m,n))

    return m,n,time_tables

print(create_time_tables())