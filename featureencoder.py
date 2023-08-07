class FeatureEncoder:
    def __init__(self):
        self.feature_function_coding = []

    def add_feature_function(self, feature_function):
        self.feature_function_coding.append(feature_function)

    def cat_count(self):
        return "cat_count"

    def cat2cat_nunique(self):
        return "cat2cat_nunique"

    def cat2num_mean(self):
        return "cat2num_mean"

    def num_sqrt(self):
        return "num_sqrt"

    def num_minmaxscaler(self):
        return "num_minmaxscaler"

    def num_log(self):
        return "num_log"

    def num_reciprocal(self):
        return "num_reciprocal"

    def num2num_add(self):
        return "num2num_add"

    def num2num_sub(self):
        return "num2num_sub"

    def num2num_mul(self):
        return "num2num_mul"

    def num2num_div(self):
        return "num2num_div"

#主函数
if __name__ == '__main__':
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

    print(encoder.feature_function_coding)