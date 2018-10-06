#coding:utf-8

# 第一步：导入各种库和包

import tensorflow as tf 

# 加载数据集MNIST
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 第二步：定义输入和输出参数

# 定义x，它是一个占位符(placeholder),代表待识别的图片
# None表示这一维的大小可以是任意的
# 每张图片用一个784维的向量表示
x = tf.placeholder(tf.float32, [None, 784])
# 定义y_，它是实际的图像标签，同样以占位符表示
y_ = tf.placeholder(tf.float32, [None, 10])

# 定义Softmax模型参数W，将一个784维的输入转换为一个10维的输出
# 在Tensorflow中，变量的参数用tf.Variable表示
W = tf.Variable(tf.zeros([784, 10]))

# 定义Softmax模型参数b，一般叫作“偏置项”(bias)
b = tf.Variable(tf.zeros([10]))

# 定义y，表示模型的输出，定义了一个Softmax回归模型
# 系统会首先获取x, W, b 的值，再去计算y的值。
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 第三步：模型的输出是y，实际的标签为y_，它们应该越相似越好。
# 在Softmax回归模型中，通常使用“交叉熵”损失来衡量这种相似性。

# 根据y和y_ 构造交叉熵损失
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))


# 第四步：优化损失，让损失减小，使用梯度下降法
# tensorflow默认会对所有变量计算梯度，这里定义了两个变量W和b，程序将
# 使用梯度下降法对W、b计算梯度并更新它们的值。
# 0.01 是梯度下降优化器的学习率，即为跨度
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 第五步：创建会话(Session)，在会话中对变量进行初始化操作
# 变量的值就是被保存在会话中
with tf.Session() as sess:
	# 初始化所有变量，分配内存
	init_op = tf.global_variables_initializer()
	sess.run(init_op)

	# 第六步：有了会话，可以对变量W，b进行优化了
	# 进行2000步梯度下降
	for i in range(2000):
		# 在mnist.train中取100个训练数据，
		# batch_xs是形状为(100,784)的图像数据，
		# batch_ys是形状为(100,10)的实际标签。
		# batch_xs, batch_ys对应着两个占位符x和y_
		batch_xs, batch_ys = mnist.train.next_batch(100)
		# 在Session中运行train_step，运行时传入占位符的值
		sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

	# 运行完梯度下降后，可以检测模型训练的结果
	# 正确的预测结果
	# tf.argmax()的功能是取出数组中最大值的下标，用来将独热表示以及模型输出转换为数字标签
	# tf.equal()函数比较它们是否相等
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	# 计算准确率，它们都是Tensor
	# 这个tf.cast()运算首先将一个布尔型的数值转换为实数型，然后计算平均值。这个平均值就是模型在这
	# 一组数据上的正确率。
	# 也就是将比价值转换成float32型的变量，此时True会被转换成1，False会被转换成0
	# tf.reduce_mean()计算数组中所有元素的平均值
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	# 这里是获取最终模型的准确率
	print("准确率：", sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))














