import tensorflow as tf
import pandas as pd
import glob
import numpy as np
import os


def read_pic():
    """
    读取图片数据
    :return:
    """
    # 构造文件名队列
    file_names = glob.glob("./leaf/*.jpg")
    # print("file_names:\n", file_names)
    file_queue = tf.train.string_input_producer(file_names)

    # 读取与解码
    reader = tf.WholeFileReader()
    filename, image = reader.read(file_queue)

    # 解码阶段
    decoded = tf.image.decode_jpeg(image)
    # print("decoded:\n", decoded)

    # 更新形状，将图片形状去定下来
    decoded.set_shape([416, 416, 3])
    # print("decoded:\n", decoded)

    # 修改图片类型
    image_cast = tf.cast(decoded, tf.float32)
    # print("image_cast:\n", image_cast)

    # 批处理
    filename_batch, image_batch = tf.train.batch([filename, image_cast], batch_size=20, num_threads=16, capacity=20)
    return filename_batch, image_batch


def parse_csv():
    """
    解析CSV
    :return:
    """
    csv_data = pd.read_csv("leaf.csv", names=["file_num", "chars"], index_col="file_num")

    labels = []

    for label in csv_data["chars"]:
        #     print(label)
        letter = []
        for word in label:
            letter.append(ord(word) - ord("A"))
        labels.append(letter)

    csv_data["labels"] = labels
    return csv_data


def filename2label(filename, csv_data):
    """
    将样本特征值和目标值一一对应
    :param filename:
    :param csv_data:
    :return:
    """
    # print(filename)
    labels = []

    for file_name in filename:
        file_num = "".join(list(filter(str.isdigit, str(file_name))))
        target = csv_data.loc[int(file_num), "labels"]
        # print(target)
        labels.append(target)
        # print(file_num)
    # print(labels)

    return np.array(labels)


def create_weights(shape):

    return tf.Variable(initial_value=tf.random_normal(shape=shape, stddev=0.01))


def create_model(x):
    """
    构建卷积神经网络
    :param x:[None, 416, 416, 3]
    :return:
    """
    # 1)第一个卷积大层
    with tf.variable_scope("conv1"):

        # 卷积层
        # 定义filter和偏置
        conv1_weight = create_weights(shape=[5, 5, 3, 32])
        conv1_bias = create_weights(shape=[32])
        conv1_x = tf.nn.conv2d(x, filter=conv1_weight, strides=[1, 1, 1, 1], padding="SAME") + conv1_bias

        # 激活层
        relu1_x = tf.nn.relu(conv1_x)

        # 池化层
        pool1_x = tf.nn.max_pool(value=relu1_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 2)第二个卷积大层
    with tf.variable_scope("conv2"):

        # 卷积层
        # 定义filter和偏置
        conv2_weight = create_weights(shape=[5, 5, 32, 64])
        conv2_bias = create_weights(shape=[64])
        conv2_x = tf.nn.conv2d(pool1_x, filter=conv2_weight, strides=[1, 1, 1, 1], padding="SAME") + conv2_bias

        # 激活层
        relu2_x = tf.nn.relu(conv2_x)

        # 池化层
        pool2_x = tf.nn.max_pool(value=relu2_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

     # 3)全连接层
    with tf.variable_scope("full_connection"):
        # [None, 104, 104, 64] --> [None, 104*104*64]
        # [None, [104*104*64]] * [104*104*64, 1*10] = [None, 1*10]
        x_fc = tf.reshape(pool2_x, shape=[-1, 104 * 104 * 64])
        weights_fc = create_weights(shape=[104 * 104 * 64, 1 * 10])
        bias_fc = create_weights(shape=[10])
        y_predict = tf.matmul(x_fc, weights_fc) + bias_fc

    return y_predict


if __name__ == "__main__":
    filename, image = read_pic()
    csv_data = parse_csv()

    # 1、准备数据
    x = tf.placeholder(tf.float32, shape=[None, 416, 416, 3])
    y_true = tf.placeholder(tf.float32, shape=[None, 1 * 10])

    # 2、构建模型
    y_predict = create_model(x)

    # 3、构造损失函数
    loss_list = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_predict)
    loss = tf.reduce_mean(loss_list)

    # 4、优化损失
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # 5、计算准确率
    equal_list = tf.reduce_all(
    tf.equal(tf.argmax(tf.reshape(y_predict, shape=[-1, 1, 10]), axis=2),
             tf.argmax(tf.reshape(y_true, shape=[-1, 1, 10]), axis=2)), axis=1)
    accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 初始化变量
    init = tf.global_variables_initializer()

    # 开启会话
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init)

        # 开启线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(1000):
            filename_value, image_value = sess.run([filename, image])
            # print("filename_value:\n", filename_value)
            # print("image_value:\n", image_value)

            labels = filename2label(filename_value, csv_data)

            # 将标签值转换成one-hot
            labels_value = tf.reshape(tf.one_hot(labels, depth=10), [-1, 1*10]).eval()
            # print(labels_value)
            _, error, accuracy_value = sess.run([optimizer, loss, accuracy], feed_dict={x: image_value,
                                                                                        y_true: labels_value})

            # print("y_true:\n", y_true.eval())
            # print("y_predict:\n", y_predict.eval())
            print("第%d次训练后损失为%f，准确率为%f" % (i+1, error, accuracy_value))

        # 回收线程
        coord.request_stop()
        coord.join(threads)
