import numpy as np
import geatpy as ea

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin):  # M 目标变量维数f的个数
        self.name = name  # 初始化name（函数名称，可以随意设置）'MyProblem'
        self.M = M #目标函数维数
        self.maxormins = maxormins  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）[1]
        # 离散变量则添加此行 self.var_set = np.array([1.1, 1, 0, 3, 5.5, 7.2,9])  # 设定一个集合，要求决策变量的值取自于该集合
        self.Dim = Dim  # 初始化Dim（决策变量维数）10
        self.varTypes = varTypes * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）[0]
        self.lb = lb * Dim  # 决策变量下界 [0]
        self.ub = ub * Dim  # 决策变量上界 [9]
        self.lbin = lbin * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）[1]
        self.ubin = ubin * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）[1]
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, self.name, self.M, self.maxormins, self.Dim, self.varTypes, self.lb, self.ub, self.lbin, self.ubin)

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

        return f

    def Evolution(self,
                 Problem=None,
                 XOVR=None,
                 Pm=None,
                 Encoding='RI',
                 NIND=None,
                 MAXGEN=None,
                 MAXTIME=None,
                 MAXEVALS=None,
                 MAXSIZE=None,
                 logTras=None,
                 prophet=None,
                 seed=None,
                 verbose=None,
                 outFunc=None,
                 drawing=None,
                 outputMsg=None,
                 drawLog=None,
                 saveFlag=None,
                 trappedValue=None,
                 maxTrappedCount=None,
                 dirName=None,
                 **kwargs):
        #构建算法
        algorithm = ea.soea_SGA_templet(
            problem,
            ea.Population(Encoding=Encoding, Field=None, NIND=NIND, Chrom=None, ObjV=None, FitnV=None, CV=None, Phen=None),
            MAXGEN,
            MAXTIME,
            MAXEVALS,
            MAXSIZE,
            logTras,
            verbose,
            outFunc,
            drawing,
            trappedValue,
            maxTrappedCount,
            dirName,
            **kwargs
            )

        algorithm.recOper.XOVR=XOVR   # 交叉概率
        algorithm.mutOper.Pm=Pm       # 变异概率
        #求解
        res = ea.optimize(algorithm,
             seed,
             prophet,
             verbose,
             drawing,
             outputMsg,
             drawLog,
             saveFlag,
             dirName,
             **kwargs)
        print(res)


if __name__ == '__main__':   #主函数
    # 实例化问题对象
    problem = MyProblem(
        name='MyProblem',  # 初始化name（函数名称，可以随意设置）'MyProblem'
        M=1,               # M (目标函数维数)
        maxormins=[1],     # maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim=10,            # Dim（决策变量维数）
        varTypes=[0],      # varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb=[0],            # lb 决策变量下界
        ub=[9],            # ub 决策变量上界
        lbin=[1],          # lbin 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin=[1]           # ubin 决策变量上边界（0表示不包含该变量的下边界，1表示包含）
        )
    # 执行进化算法
    MyProblem.Evolution(
        problem,
        XOVR = 0.7,          # 交叉概率
        Pm = 0.05,           # 变异概率
        Encoding = 'RI',     # str - 染色体编码方式，'BG':二进制/格雷编码；'RI':实整数编码，即实数和整数的混合编码；'P':排列编码。
        NIND = 30,           # int - 种群染色体长度。
        MAXGEN = 300,        # 最大进化代数
        MAXTIME = None,      # 最长迭代时间
        MAXEVALS = None,
        MAXSIZE = None,      #
        logTras = 0,         # 表示每隔多少代记录一次日志信息，0表示不记录。
        prophet = None,      # <class: Population> / Numpy ndarray - 先验知识。可以是种群对象，也可以是一组或多组决策变量组成的矩阵(矩阵的每一行对应一组决策变量)
        seed = None,         # int - 随机数种子。
        verbose = False,     # bool - 控制是否在输入输出流中打印输出日志信息。该参数将被传递给algorithm.verbose。
                             # 如果algorithm已设置了该参数的值，则调用optimize函数时，可以不传入该参数。
        outFunc = None,
        drawing = 1,         # int - 算法类控制绘图方式的参数，0表示不绘图；1表示绘制最终结果图；2表示实时绘制目标空间动态图；
                             # 3表示实时绘制决策空间动态图。该参数将被传递给algorithm.drawing。如果algorithm已设置了该参数的值，则调用optimize函数时，可以不传入该参数。
        outputMsg = True,    # bool - 控制是否输出结果以及相关指标信息。
        drawLog = False,     # bool - 用于控制是否根据日志绘制迭代变化图像。
        saveFlag = True,     # bool - 控制是否保存结果
        trappedValue = None,
        maxTrappedCount = None,
        dirName = None,      # str - 文件保存的路径。当缺省或为None时，默认保存在当前工作目录的'result of job xxxx-xx-xx xxh-xxm-xxs'文件夹下。
        )


