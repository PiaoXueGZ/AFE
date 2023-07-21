#约束进化算法
import geatpy as ea
import numpy as np

class GA_CNN:
    def __init__(self, args, cfg, log):     #cfg库 文件读取 log库 调取电脑
        self.args = args
        self.log = log
        self.cfg = cfg

        '''GA_params'''
        self.MAXGEN = self.cfg.PARA.GA_params.MAXGEN        #最大迭代次数
        self.Nind = self.cfg.PARA.GA_params.Nind            #种群数量
        self.maxormins = self.cfg.PARA.GA_params.maxormins  # -1：最大化 1：最小化
        self.xov_rate = self.cfg.PARA.GA_params.xov_rate    # 交叉概率

        '''chrom'''
        self.params_dict = {}
        self.FieldDR = None             #译码矩阵
        self.chrom_all = None           #染色体  X
        self.Objv_all = None            #目标 Y


        '''记录每一代的数据'''
        self.obj_trace = np.zeros((self.MAXGEN, 2))  # [MAXGEN, 2] 其中[0]记录当代种群的目标函数均值，[1]记录当代种群最优个体的目标函数值
        self.var_trace = np.zeros((self.MAXGEN, self.cfg.PARA.CNN_params.CNN_total_params))  # 记录当代种群最优个体的变量值
        self.time = None

        '''记录所有种群中的最优值'''
        self.best_gen = None
        self.best_Objv = None
        self.best_chrom_i = None
        # self.X = None # 把X和Y用chrom_all和Objv_all代替
        # self.Y = None

    #种群初始化
    def Init_chrom(self):
        # 决策变量X的上界
        lb = np.hstack((
            self.cfg.PARA.CNN_params.conv_channels_1[0],
            self.cfg.PARA.CNN_params.conv_channels_2[0],
            self.cfg.PARA.CNN_params.conv_kernel_size[0],
            self.cfg.PARA.CNN_params.conv_kernel_size[0],
            self.cfg.PARA.CNN_params.pool_kernel_size[0],
            self.cfg.PARA.CNN_params.pool_stride[0],
            self.cfg.PARA.CNN_params.fc_features_1[0],
            self.cfg.PARA.CNN_params.fc_features_2[0],
        ))
        # 决策变量X的下界
        ub = np.hstack((
            self.cfg.PARA.CNN_params.conv_channels_1[1],
            self.cfg.PARA.CNN_params.conv_channels_2[1],
            self.cfg.PARA.CNN_params.conv_kernel_size[1],
            self.cfg.PARA.CNN_params.conv_kernel_size[1],
            self.cfg.PARA.CNN_params.pool_kernel_size[1],
            self.cfg.PARA.CNN_params.pool_stride[1],
            self.cfg.PARA.CNN_params.fc_features_1[1],
            self.cfg.PARA.CNN_params.fc_features_2[1],
        ))
        varTypes = [1] * (self.cfg.PARA.CNN_params.CNN_total_params) #0表示决策变量为连续型，1表示为离散型
        FieldDR = np.vstack((lb, ub, varTypes))
        chrom = ea.crtip(self.Nind, FieldDR)  #创建元素是整数的种群染色体矩阵

        self.FieldDR = FieldDR  # 固定不变的

        return chrom

    #种群目标函数值矩阵
    def get_Objv_i(self, chrom):  #输入的chrom个体数 = 每一代种群的个数Nind
        chrom = np.int32(chrom)   #整数范围（-2147483648 to 2147483647）
        num = chrom.shape[0]

        Objv = np.zeros(shape=(num, 1)) #创建num行1列的列向量
        for i in range(num):
            self.params_dict['conv_channels'] = chrom[i, 0:2] #前两个，每次都会更新，重新赋值
            self.params_dict['conv_kernel_size'] = chrom[i, 2:4]
            self.params_dict['pool_kernel_size'] = chrom[i, 4]
            self.params_dict['pool_stride'] = chrom[i, 5]
            self.params_dict['fc_features'] = chrom[i, 6:8]
            Objv[i] = Train_GACNN(self.params_dict, self.args, self.cfg, self.log)

        # Objv = np.random.rand(num, 1)

        return Objv

    def Evolution(self): #
        start_time = time.time()

        # 初始化种群
        Init_chrom = self.Init_chrom()

        # 开始进化
        self.log.logger.info('==> This is Init GEN <==' )
        Init_Objv = self.get_Objv_i(Init_chrom)
        best_ind = np.argmax(Init_Objv * self.maxormins) #记录最优个体的索引值

        self.chrom_all = Init_chrom
        self.Objv_all = Init_Objv

        for gen in range(self.MAXGEN):
            self.log.logger.info('==> This is No.%d GEN <==' % (gen))

            if gen==0: #第一代和后面有所不同
                chrom = Init_chrom
                Objv = Init_Objv

            else:
                chrom = NewChrom
                Objv = NewObjv

            FitnV = ea.ranking(Objv * self.maxormins)   #计算种群适应度
            Selch = chrom[ea.selecting('rws', FitnV, self.Nind-1), :] #轮盘赌选择 Nind-1 代，与上一代的最优个体再进行拼接
            Selch = ea.recombin('xovdp', Selch, self.xov_rate) #重组，采取两点交叉
            Selch = ea.mutate('mutuni', 'RI', Selch, self.FieldDR) #变异
            Objv_Selch = self.get_Objv_i(Selch)

            NewChrom = np.vstack((chrom[best_ind, :], Selch)) #将上一代的最优个体与现在的种群拼接
            NewObjv =np.vstack((Objv[best_ind, :], Objv_Selch))
            best_ind = np.argmax(NewObjv * self.maxormins)

            self.chrom_all = np.vstack((self.chrom_all, NewChrom))
            self.Objv_all = np.vstack((self.Objv_all, NewObjv))

            self.obj_trace[gen, 0] = np.sum(NewObjv) / self.Nind  # 记录当代种群的目标函数均值
            self.obj_trace[gen, 1] = NewObjv[best_ind]  # 记录当代种群最有给他目标函数值
            self.var_trace[gen, :] = NewChrom[best_ind, :]  # 记录当代种群最优个体的变量值
            self.log.logger.info('GEN=%d,best_Objv=%.5f,best_chrom_i=%s\n'
                                 % (gen, NewObjv[best_ind], str(NewChrom[best_ind, :])))  # 记录每一代的最大适应度值和个体

        self.Save_chroms(self.chrom_all)
        self.Save_objvs(self.Objv_all)

        end_time = time.time()
        self.time = end_time - start_time
        self.log.logger.info('The time of Evoluation is %.5f s. ' % self.time)

    def Plot_Save(self):
        self.best_gen = np.argmax(self.obj_trace[:, [1]])
        self.best_Objv = self.obj_trace[self.best_gen, 1]
        self.best_chrom_i = self.var_trace[self.best_gen]

        # pdb.set_trace()
        ea.trcplot(trace=self.obj_trace,
                   labels=[['POP Mean Objv', 'Best Chrom i Objv']],
                   titles=[['Mean_Best_Chromi_Objv']],
                   save_path=self.cfg.PARA.CNN_params.save_data_path,
                   xlabels=[['GEN']],
                   ylabels=[['ACC']])

        with open(self.cfg.PARA.CNN_params.save_bestdata_txt, 'a') as f:
            f.write('best_Objv=%.5f,best_chrom_i=%s,total_time=%.5f\n' % (
            self.best_Objv, str(self.best_chrom_i), self.time))

        np.savetxt(self.cfg.PARA.CNN_params.save_data_path + 'MeanChromi_Objv.txt', self.obj_trace[:, 0])
        np.savetxt(self.cfg.PARA.CNN_params.save_data_path + 'BestChromi_Objv.txt', self.obj_trace[:, 1])
        np.savetxt(self.cfg.PARA.CNN_params.save_data_path + 'BestChromi.txt', self.var_trace)

    def Save_chroms(self, chrom):
        self.log.logger.info('==> Save Chroms to file <==')
        scio.savemat(self.cfg.PARA.CNN_params.save_x_mat, {"chrom": chrom})

    def Save_objvs(self, Objv):
        self.log.logger.info('==> Save Objvs to file <==')
        scio.savemat(self.cfg.PARA.CNN_params.save_y_mat, {"objv": Objv})