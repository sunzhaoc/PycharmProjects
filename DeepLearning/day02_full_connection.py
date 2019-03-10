import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data


# 1、利用数据，在训练的时候实时提供数据
# mnist手写数字数据在运行时候实时提供给占位符

tf.app.flags.DEFINE_integer("is_train", 1, "指定是否训练模型，还是拿数据去预测")
FLAGS = tf.app.flags.FLAGS


def full_connection():
    """
    单层全连接神经网络识别手写熟悉图片
    特征值:[None, 784]
    目标值:[None, 10]
    :return:
    """
    mnist = input_data.read_data_sets("./mnist_data", one_hot=True)
    # 1、准备数据
    with tf.variable_scope("mnist_data"):
        x = tf.placeholder(dtype=tf.float32, shape=(None, 784))
        y_true = tf.placeholder(dtype=tf.float32, shape=(None, 10))

    # 2、全连接层神经网络计算
    # 类别：10个类别 全连接层：10个神经元
    # 参数w:[784, 10]    b:[10]
    # 全连接层神经网络的计算公式:[None, 784] * [784, 10] + [10] = [None, 10]
    # 随机初始化权重偏置参数，这些事优化的参数，必须要使用变量op去定义
    with tf.variable_scope("fc_model"):
        Weight = tf.Variable(initial_value=tf.random_normal(shape=[784, 10]))
        bias = tf.Variable(initial_value=tf.random_normal(shape=[10]))
        # fc层的计算
        y_predict =tf.matmul(x, Weight) + bias

    # 3、softmax回归以及交叉熵损失计算
    with tf.variable_scope("softmax_crossentropy"):
        # label：真实值 [None, 10] one_hot
        # logits：全连接层的输出[None, 10]
        # 返回每个样本的损失组成的列表
        error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    # 4、梯度下降损失优化
    with tf.variable_scope("optimizer"):
        # 学习率
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(error)

    # 5、得出每次训练的准确率（通过真实值和预测值进行位置比较，每个样本都比较）
    with tf.variable_scope("accuracy"):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # (2)收集要显示的变量
    # 先收集损失和准确率
    tf.summary.scalar("error", error)
    tf.summary.scalar("acc", accuracy)
    # 收集权重和偏置
    tf.summary.histogram("weights", Weight)
    tf.summary.histogram("bias", bias)

    # 初始化变量op
    init = tf.global_variables_initializer()

    # (3)合并所有变量op
    merged = tf.summary.merge_all()

    # 创建模型保存和加载
    saver = tf.train.Saver()

    # 开启会话去训练
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init)

        # (1)创建一个events文件实例
        file_writer = tf.summary.FileWriter("./tmp/summary", graph=sess.graph)

        # 加载模型
        if os.path.exists("./tmp/modelckpt/checkpoint"):
            saver.restore(sess, "./tmp/modelckpt/fc_nn_model")

        if FLAGS.is_train == 1:
            # 循环步数去训练
            # 每步提供100个样本
            mnist_x, mnist_y = mnist.train.next_batch(100)

            for i in range(500):
                # 获取数据，实时提供
                # 运行训练op
                sess.run(optimizer, feed_dict={x: mnist_x, y_true: mnist_y})
                print("训练第%d步的准确率为：%f，损失为：%f" % (i+1,
                                                 sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y}),
                                                 sess.run(error, feed_dict={x: mnist_x, y_true: mnist_y})
                                                 )
                      )
                # 运行合变量op，写入事件文件当中
                summary = sess.run(merged, feed_dict={x: mnist_x, y_true: mnist_y})
                file_writer.add_summary(summary, i)
                if i % 100 == 0:
                    saver.save(sess, "./tmp/modelckpt/fc_nn_model")

        else:
            # 如果不是训练，我们就去进行预测测试数据集
            for i in range(100):
                # 每次拿一个样本预测
                mnist_x, mnist_y = mnist.test.next_batch(1)
                print("第%d个样本的真实值为：%d，模型预测结果为：%d" % (i+1, tf.argmax(sess.run(y_true, feed_dict={x: mnist_x, y_true: mnist_y}), tf.argmax(sess.run(y_predict, feed_dict={x: mnist_x, y_true: mnist_y})))))
        # image, label = mnist.train.next_batch(100)
        #
        # print("训练之前，损失为%f" % sess.run(error, feed_dict={x: image, y_true: label}))
        #
        # # 开始训练
        # for i in range(500):
        #     _, loss, accuracy_value = sess.run([optimizer, error, accuracy], feed_dict={x: image, y_true: label})
        #     print("第%d次的训练，损失为%f，准确率为%f" % (i+1, loss, accuracy_value))

    return None


if __name__ == "__main__":
    full_connection()
