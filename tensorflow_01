# coding:utf-8
# 建立简单神经网络
import tensorflow as tf

# 输入数据 (常量数据，1x2 的 ， 体积和重量)
x = tf.constant([[0.7, 0.5]])
# 权重参数 
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# 定义前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
# 用会话计算结果(数据通过神经网络后生成一个结果，可理解为数据的特征值)
with tf.Session() as sess:
    init = tf.global_variables_initializer() 
    sess.run(init)
    print("y : %s" %sess.run(y))
    
    
# 输入数据 (使用placeholder 占位，进行动态传入数据) shape=(1, 2) 传入1行2列数据
x = tf.placeholder(tf.float32, shape=(1, 2))
# 权重参数 
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# 定义前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
# 用会话计算结果(数据通过神经网络后生成一个结果，可理解为数据的特征值)
with tf.Session() as sess:
    init = tf.global_variables_initializer() 
    sess.run(init)
    #   这里要动态传入数据值
    print("y : %s" %sess.run(y, feed_dict={x: [[0.7, 0.5]]}))
    
    
# 输入数据 （使用placeholder占位，shape=（None，2）输入多组数据）
x = tf.placeholder(tf.float32, shape=(None, 2))
# 设置权重参数 输入数据为2 ，输出为3， 偏差值为1， 随机种子为1
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# 添加神经层
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
# 使用会话进行运算zip 注意动态输入多组数据
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print("y : %s" %sess.run(y, feed_dict={x:[[0.7, 0.5], [0.8, 0.6], [0.3, 0.2]]}))
