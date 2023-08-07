from MyProblem import MyProblem  # 导入自定义问题接口
import geatpy as ea  # import geatpy

if __name__ == '__main__':
    # 实例化问题对象
    problem = MyProblem()
    # 构建算法
    algorithm = ea.soea_SGA_templet(
        problem,
        ea.Population(Encoding='RI', NIND=30),
        MAXGEN=300,  # 最大进化代数
        logTras=0)  # 表示每隔多少代记录一次日志信息，0表示不记录。

    algorithm.recOper.XOVR=0.7  #交叉概率
    algorithm.mutOper.Pm=0.05   #变异概率
    # 求解

    res = ea.optimize(algorithm,verbose=False,drawing=1,outputMsg=True,drawLog=False,saveFlag=True)
    print(res)