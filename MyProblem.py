import numpy as np
import geatpy as ea
# min f=（x0-0)**2+（x1-1)**2+（x2-2)**2+（x3-3)**2+（x4-4)**2+（x5-5)**2+（x6-6)**2+（x7-7)**2+（x8-8)**2+（x9-9)**2
#x0,x1...x8,x9<=9
#x0,x1...x8,x9>=0


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=1):  #M 目标变量维数f的个数
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        maxormins = [1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        # 离散变量则添加此行 self.var_set = np.array([1.1, 1, 0, 3, 5.5, 7.2,9])  # 设定一个集合，要求决策变量的值取自于该集合
        Dim = 10  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [0] * Dim  # 决策变量下界
        ub = [9] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def evalVars(self, Vars):  # 目标函数
        x0 = Vars[:, [0]]
        x1 = Vars[:, [1]]
        x2 = Vars[:, [2]]
        x3 = Vars[:, [3]]
        x4 = Vars[:, [4]]
        x5 = Vars[:, [5]]
        x6 = Vars[:, [6]]
        x7 = Vars[:, [7]]
        x8 = Vars[:, [8]]
        x9 = Vars[:, [9]]
        f = (x0 - 0)**2 +(x1 - 1) ** 2 +(x2 - 2) ** 2 +(x3 - 3) ** 2 +(x4 - 4)**2 +(x5 - 5) ** 2 +(x6 - 6) ** 2 +(x7 - 7) ** 2 +(x8 - 8) ** 2 +(x9 - 9) ** 2

        # 利用可行性法则处理约束条件   此例可不要CV
        # CV = np.hstack([
        #    9 - x0, 9 - x1, 9 - x2, 9 - x3, 9 - x4, 9 - x5, 9 - x6, 9 - x7, 9 - x8, 9 - x9,
        #    -x1, -x2, -x3, -x4, -x5, -x6, -x7, -x8, -x9
        # ])

        return f