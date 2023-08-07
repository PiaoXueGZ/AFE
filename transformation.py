import numpy as np
from Evolution2 import EAAFE
from featureencoder import FeatureEncoder

class FeatureTransformer:
    def cat_count(self, arr):
        #依次统计数组中每个元素出现的次数，用次数替换原来的元素
        arr2 = np.zeros_like(arr)
        for i in range(len(arr)):
            arr2[i] = np.sum(arr == arr[i])
        return arr2
    
    def cat2cat_nunique(self, arr1, arr2):
        combined = [tuple(map(str, pair)) for pair in zip(arr1, arr2)]
        return np.array([len(np.unique(c)) for c in combined])
    
    def cat2num_mean(self, arr_cat, arr_num):
        unique_cats = np.unique(arr_cat)
        cat_means = {}
        for cat in unique_cats:
            cat_indices = np.where(arr_cat == cat)
            cat_means[cat] = np.mean(arr_num[cat_indices])
        return np.array([cat_means[cat] for cat in arr_cat])
    
    def num_sqrt(self, arr):
        #取绝对值避免出现负数
        arr = np.abs(arr)
        return np.sqrt(arr)
    
    def num_minmaxscaler(self, arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        #避免出现除数为0
        if max_val == min_val:
            #返回全1数组
            return np.ones_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    def num_log(self, arr):
        #取绝对值避免出现负数
        arr = np.abs(arr)
        arr = arr + 1e-3
        return np.log(arr)
    
    def num_reciprocal(self, arr):
        #避免出现0
        arr = arr + 1e-3
        return 1.0 / arr
    
    def num2num_add(self, arr1, arr2):
        return arr1 + arr2
    
    def num2num_sub(self, arr1, arr2):
        return arr1 - arr2
    
    def num2num_mul(self, arr1, arr2):
        return arr1 * arr2
    
    def num2num_div(self, arr1, arr2):
        #避免除数为0
        arr2 = arr2 + 1e-3
        return arr1 / arr2
    
    def data_transform(self, chrom_j, data):
        eaafe = EAAFE()
        encoder = FeatureEncoder()
        encoder.add_feature_function(encoder.cat2num_mean())
        encoder.add_feature_function(encoder.cat_count())
        encoder.add_feature_function(encoder.cat2cat_nunique())
        encoder.add_feature_function(encoder.num_sqrt())
        encoder.add_feature_function(encoder.num_minmaxscaler())
        encoder.add_feature_function(encoder.num_log())
        encoder.add_feature_function(encoder.num_reciprocal())
        encoder.add_feature_function(encoder.num2num_add())
        encoder.add_feature_function(encoder.num2num_sub())
        encoder.add_feature_function(encoder.num2num_mul())
        encoder.add_feature_function(encoder.num2num_div())
        for gen in range(eaafe.gen_num):
            arr = data[:,gen]
            for tran in range(eaafe.tran_num):
                num = chrom_j[gen*3+tran]
                func_name = encoder.feature_function_coding[num]
                func = getattr(self, func_name)
                if num >= 3 and num <= 6:
                    arr = func(arr)
                elif num >= 7 and num <= 10:
                    if gen == eaafe.gen_num - 1:
                        arr = func(arr, arr)
                    else:
                        arr = func(arr, data[:,gen+1])

                #将arr放入data最后一列
            data = np.c_[data, arr]

        return data


if __name__ == "__main__":
    transformer = FeatureTransformer()
    cat_feature = np.array(['A', 'B', 'A', 'C', 'B'])
    num_feature = np.array([1, 2, 1, 4, 5])
    
    print(transformer.cat_count(num_feature))
    # 输出: [{'A': 2, 'B': 2, 'C': 1}]

    print(transformer.cat2cat_nunique(cat_feature, cat_feature))
    # 输出: [2, 2, 2, 1, 2]
    
    print(transformer.cat2num_mean(cat_feature, num_feature))
    # 输出: [2.0, 3.5, 2.0, 4.0, 3.5]
    
    print(transformer.num_sqrt(num_feature))
    # 输出: [1.         1.41421356 1.73205081 2.         2.23606798]
    
    print(transformer.num_minmaxscaler(num_feature))
    # 输出: [0.   0.25 0.5  0.75 1.  ]
    
    print(transformer.num_log(num_feature))
    # 输出: [0.         0.69314718 1.09861229 1.38629436 1.60943791]
    
    print(transformer.num_reciprocal(num_feature))
    # 输出: [1.         0.5        0.33333333 0.25       0.2       ]
    
    print(transformer.num2num_add(num_feature, num_feature))
    # 输出: [ 2  4  6  8 10]
    
    print(transformer.num2num_sub(num_feature, num_feature))
    # 输出: [0 0 0 0 0]
    
    print(transformer.num2num_mul(num_feature, num_feature))
    # 输出: [ 1  4  9 16 25]
    
    print(transformer.num2num_div(num_feature, num_feature))
    # 输出: [1. 1. 1. 1. 1.]
