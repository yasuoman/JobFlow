# project : job_flow_1
# file   : F2_Johnson.py
# author:yasuoman
# datetime:2020/8/5 10:22
# software: PyCharm

import numpy as np
"""
description：
说明：实现两台机器的Johnson规则,找到最优解
返回的是最优解的时间和相应的处理工件的顺序
"""
def F2_Johnson_optimal_solution(M1,M2,n):
    c = n * ['']  # 加工次序列表
    node_dict = {}  # 实例化对象字典
    for i in range(n):
        node_dict[i] = Node()

    first = 0
    end = n - 1
    for i in range(n):
        node_dict[i].index = i  # 记录一下当前这个node节点放的是哪个作业
        if M1[i] > M2[i]:
            # 后行工序
            node_dict[i].time = M2[i]
            node_dict[i].position = 2
        else:
            # 先行工序
            node_dict[i].time = M1[i]
            node_dict[i].position = 1

        # 虽然把n个作业都赋值到了Node型结构体中，
        # 但是大小交错，没有顺序，需要根据每个对象的time值从小到大排序
        # 所以需要排序
    node_dict = dict(sorted(node_dict.items(), key=lambda x: x[1].time, reverse=False))

    # print(node_dict.values())
    for value in node_dict.values():
        # 先行
        if value.position == 1:
            c[first] = value.index
            first = first + 1
        # 后行
        if value.position == 2:
            c[end] = value.index
            end = end - 1
    # 分别记录在机器1 和 机器2 上的时间
    time1 = M1[c[0]]
    time2 = time1 + M2[c[0]]
    for i in range(1, n):
        time1 = time1 + M1[c[i]]
        if time1 > time2:
            time2 = time1 + M2[c[i]]
        else:
            time2 = time2 + M2[c[i]]

    return time2,np.array(c)+1

class Node:
    def __init__(self):
        self.time=0  #时间
        self.index=0  #来自第几个作业
        self.position=0 #是先行工序还是后行工序


M1=[50, 18, 96, 26, 14, 48, 80, 32, 59, 15, 30, 75, 44, 66, 64, 19,
        34, 47, 47, 83]
M2=[27, 45, 54, 68, 92, 97, 71, 11, 30, 93, 19, 63, 65, 88, 79, 57,
        86, 34, 57, 73]

# print(F2_Johnson_optimal_solution(M1,M2,20))





