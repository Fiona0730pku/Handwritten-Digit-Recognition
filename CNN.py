#第一个cnn神经网络
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

def compute_accuracy(v_xs,v_ys):
	global prediction
	y_pre=sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
	correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
	accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
	return result
	
def weight_variable(shape):
	#输入shape返回variable
	initial=tf.truncated_normal(shape,stddev=0.1) #standard deviation
	#类似normal distribution 但在卷积神经网络中更好用 google就是用这个来产生随机变量
	return tf.Variable(initial)

def bias_variable(shape):
	initial=tf.constant(0.1,shape=shape) #初始值让所有都为0.1 bias通常设为正值会比较好
	return tf.Variable(initial)
	
def conv2d(x,W): #x为输入图片的值 W为前面生成的Weight->是作为cnn的weight
	#2维的cnn
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')#strides移动间隔 是四个长度的列表
	#头尾均为1(tensorflow规定) 中间两个元素分别为长度为x,y方向的移动间隔
	#stride[1,x_movement,y_movement,1]

def max_pool_2x2(x): #x是从conv层里面传出来的东西
	#为了防止跨度(stride)太大丢失信息 把跨度减小 再用pooling来进行类似于把跨度变大的操作 结果得到的图片大小一样 但保留了更多的信息
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
	#ksize 池化窗口的大小 四维向量 一般为[1,height,width,1]
	#pooling的stride需要比conv的stride更大一些 起到压缩图片长宽的作用(这次是两步一移动)
	#pooling和conv很像 但是区别在于pooling不需要传入weight
	
xs=tf.placeholder(tf.float32,[None,784]) #28*28
ys=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)
#把图片信息换成另一种信息
x_image=tf.reshape(xs,[-1,28,28,1])
#xs是所有图片的信息 -1表示先不管维度 最后再加上维度(可以理解为导入数据有多少图片 这要最后才知道)
#28*28像素点 channel=1 因为这里是黑白图片 彩色则为3
#print(x_image.shape) #[n_samples,28,28,1]

#conv1 layer
W_conv1=weight_variable([5,5,1,32]) #patch 5*5,in size 1(输入图片的高度 黑白->1), out size 32(输出图片的高度) 32=28+5-1
b_conv1=bias_variable([32]) #bias有32个长度 和图片的out size相对应
#hidden convolutional layer 1   relu:在输出前再加入一个非线性处理 让它变得非线性化
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1) #和之前学的形式有些像 只是把矩阵相乘换成了经过一层卷积
#output size 28*28*32(因为选择的是same padding 所以长宽不变 只有高是变的)
h_pool1=max_pool_2x2(h_conv1) #整个layer的输出值
#output size 14*14*32(因为池化里是每两步一移)

#conv2 layer(在了解了conv1图片形状变化的基础上改动各个size)
W_conv2=weight_variable([5,5,32,64]) #把高度从32变成64 不断变高
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)	#output size 14*14*64
h_pool2=max_pool_2x2(h_conv2)	#output size 7*7*64

#func1 layer(就是之前定义的神经网络)
W_fc1=weight_variable([7*7*64,1024]) #让输出的高变为1024
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
#把pooling的结果从三维数据变成一维数据 [n_samples,7,7,64]->>[n_samples,7*7*64]
#hidden fully connected layer 1
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1) #注意是矩阵相乘 h_pool2_flat和W_fc1的顺序不能调换
#可以把h_fc1作为fc层的输出结果 也可以再加一个dropout函数
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob) #output size [n_samples,1024]

#func2 layer
W_fc2=weight_variable([1024,10]) #数字识别任务最终的输出是10位向量
b_fc2=bias_variable([10])
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2) #output size [n_samples,10]
#在输出层用softmax来进行classification的处理
#不要复制黏贴还保持着relu!!这样无论运行多少次都还是0.098!!
#选择适当的activation function可以说是非常重要了QAQ

cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #对于庞大的系统用AdamOptimizer会比较好 注意它需要更小的learning rate

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
	batch_xs,batch_ys=mnist.train.next_batch(100)
	sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
	if i%50==0:
		print(compute_accuracy(mnist.test.images,mnist.test.labels))