import numpy as np
import random

class EAAFE:
    def __init__(self):

        # 种群参数
        self.MAXGEN = 5000  # 最大迭代次数
        self.Nind = 50  # 种群数量
        self.maxormins = -1  # -1：最大化 1：最小化
        self.variation_rate = 0.1  # 变异概率
        self.xov_rate = 0.9  # 交叉概率

        # 染色体参数
        self.gen_num = 10  # 基因总数
        self.FieldDR = None  # 译码矩阵
        self.chrom_all = None  # 总染色体

        # 记录所有种群中的最优值
        self.best_fit = float('-inf')
        self.best_gen = None
        self.best_chrom_i = None

        # 基因参数
        # 决策变量X的上界
        self.lb = []
        # 决策变量X的下界
        self.ub = []
        # 决策变量是否包含上界,1为<=   0为<
        self.lbmin = []
        # 决策变量是否包含上界
        self.ubmin = []

    # 默认下界包含，上界不包含
    def init_chrom(self, lb, ub, lbmin, ubmin):

        # 决策变量X的下界
        self.lb = lb
        # 决策变量X的上界
        self.ub = ub
        # 决策变量是否包含下界,1为<=   0为<
        self.lbmin = lbmin
        # 决策变量是否包含上界
        self.ubmin = ubmin
        chrom = []  # 创建元素是整数的种群染色体矩阵

        for i in range(self.Nind):
            single_chrom = np.random.randint(lb[i], ub[i], (self.gen_num))
            chrom.append(single_chrom)

        return chrom

    def evolution(self, pre_chrom):
        chrom = pre_chrom
        for i in range(self.MAXGEN):
            live_chrom = self.rws(chrom)
            new_chrom = self.crossover(live_chrom)
            chrom = self.mutation(new_chrom)
            fitness = self.get_fitness(chrom)
            m_fit, index = np.max(fitness), np.argmax(fitness)
            if m_fit > self.best_fit:
                self.best_fit = m_fit
                self.best_chrom_i = index
                self.best_gen = self.decode(chrom[index])

    # rws函数是轮盘赌选择算法
    def rws(self, pre_chrom):
        chrom = []
        fitnesses = self.get_fitness(pre_chrom)
        fitness_sum = sum(fitnesses)
        probs = []
        for fitness in fitnesses:
            probs.append(fitness / fitness_sum)

        addedIndexes = []
        for i in range(len(fitnesses)):
            rate = random.random()
            if rate < probs[i]:
                chrom.append(pre_chrom[i])
                addedIndexes.append(i)

        #如果轮盘赌选择后的种群数量小于原来的十分之一，则将逐步加入最优个体直到数量达到原来的十分之一
        if len(chrom) < self.Nind / 10:
            sortedIndexes = np.argsort(fitnesses)
            for index in sortedIndexes:
                if index not in addedIndexes:
                    chrom.append(pre_chrom[index])
                    addedIndexes.append(index)
                if len(chrom) >= self.Nind / 10:
                    break

        return chrom

    # crossover函数是单点交叉算法
    def crossover(self, pre_chrom):
        new_chrom = []
        num = 0
        while num < self.Nind:
            chrom = pre_chrom
            a = random.randint(0, len(chrom) - 1)
            b = random.randint(0, len(chrom) - 1)
            rate = random.random()
            if rate < self.xov_rate:
                xov_point = random.randint(0, self.gen_num - 1)
                chrom[a][xov_point], chrom[b][xov_point] = chrom[b][xov_point], chrom[a][xov_point]
                if xov_point < self.gen_num - 1:
                    chrom[a][xov_point + 1], chrom[b][xov_point + 1] = chrom[b][xov_point + 1], chrom[a][xov_point + 1]

            new_chrom.append(chrom[a])
            num += 1
            if num < self.Nind:
                new_chrom.append(chrom[b])
                num += 1

        return new_chrom

    # mutation函数是变异算法
    def mutation(self, pre_chrom):
        chrom = pre_chrom
        for single_chrom in chrom:
            rate = random.random()
            if rate < self.variation_rate:
                mutation_point = random.randint(0, self.gen_num - 1)
                mutation_gen = random.randint(self.lb[mutation_point], self.ub[mutation_point])
                single_chrom[mutation_point] = mutation_gen

        return chrom

    def get_fitness(self, pre_chrom):
        chrom = pre_chrom
        fitness = []
        for single_chrom in chrom:
            single_fitness = 10  # 应为计算表达式
            # target = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            # single_fitness = -np.linalg.norm(target - single_chrom)
            fitness.append(single_fitness)

        return fitness

    # 表现型转基因型,具体问题分析
    def decode(self, chrom):
        return chrom
