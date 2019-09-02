import functools


class Perceptron(object):
    def __init__(self, input_num, activator):
        '''
        初始化感知器，设置输入参数的个数，以及激活函数。
        激活函数的类型为double -> double
        '''
        self.activator = activator
        # 权重向量初始化为0
        self.weights = [0.0 for _ in range(input_num)]
        # 偏置项初始化为0
        self.bias = 0.0

    def __str__(self):
        '''
        打印学习到的权重、偏置项
        '''
        return 'weights\t:%s\tbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_vec):
        '''
        输入向量，输出感知器的计算结果
        '''
        # 把input_vec[x1,x2,x3...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用map函数计算[x1*w1, x2*w2, x3*w3]
        # 最后利用reduce求和
        package = list(zip(input_vec, self.weights))

        map_resualt = list(map(lambda x: x[0] * x[1], package))

        reduce_resualt = functools.reduce(lambda a, b: a + b, map_resualt, 0.0)

        return self.activator(reduce_resualt + self.bias)

    def train(self, input_vecs, labels, iteration, rate):
        '''
        输入训练数据：一组向量、与每个向量对应的label；以及训练轮数、学习率
        '''
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):
        '''
        一次迭代，把所有的训练数据过一遍
        '''
        # 把输入和输出打包在一起，成为样本的列表[(input_vec, label), ...]
        # 而每个训练样本是(input_vec, label)
        samples = zip(input_vecs, labels)
        # 对每个样本，按照感知器规则更新权重
        for (input_vec, label) in samples:
            # 计算感知器在当前权重下的输出
            output = self.predict(input_vec)
            # 更新权重
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        '''
        按照感知器规则更新权重
        '''
        # 把input_vec[x1,x2,x3,...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用感知器规则更新权重
        delta = label - output
        self.weights = list(map(lambda x: x[1] + rate * delta * x[0], list(zip(input_vec, self.weights))))
        # 更新bias
        self.bias += rate * delta
        print("weights:", self.weights, end="\t")
        print("bias:", self.bias)
        
#-------------------------------------------------------------------------------------------------------------------        


def f(x):
    '''
     定义激活函数
    :param x:
    :return:
    '''
    return 1 if x > 1 else 0


def get_trainset():
    '''
    准备训练数据
    :return:
    '''
    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    labels = [1, 0, 0, 0]
    return input_vecs, labels


def train_and_Perceptron():
    '''
    传入数据训练10轮
    :return:
    '''
    # 创建感知器
    p = Perceptron(2, f)
    # 训练迭代10轮
    for i in range(10):
        input_vecs, labels = get_trainset()
        p.train(input_vecs, labels, 10, 0.05)
    # 返回训练好的感知器
    return p


if __name__ == '__main__':
    # 训练
    and_perception = train_and_Perceptron()
    # 打印权重
    print(and_perception)
    print('训练完毕')
    # 测试
    print('1 and 1 = %d' % and_perception.predict([1, 1]))
    print('0 and 0 = %d' % and_perception.predict([0, 0]))
    print('1 and 0 = %d' % and_perception.predict([1, 0]))
    print('0 and 1 = %d' % and_perception.predict([0, 1]))
