import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def tensorflow_demo():
    """
    TensorFlow的基本结构
    :return:
    """
    # 原生python加法运算
    a = 2
    b = 3
    c = a + b
    print("普通加法运算的结果：\n", c)

    # TensorFlow实现加法运算
    a_t = tf.constant(2)
    b_t = tf.constant(3)
    c_t = a_t + b_t
    print("TensorFlow加法运算的结果：\n", c_t)

    # 开启会话
    with tf.Session() as sess:
        c_t_value = sess.run(c_t)
        print("c_t_value:\n", c_t_value)

    return None


def graph_demo():
    """
    图的演示
    :return:
    """
    # TensorFlow实现加法运算
    a_t = tf.constant(2, name="a_t")
    b_t = tf.constant(3, name="b_t")
    c_t = tf.add(a_t, b_t, name="c_t")
    print("a_t:\n", a_t)
    print("b_t:\n", b_t)
    print("c_t：\n", c_t)

    # 查看默认图
    # 方法1：调用方法
    default_g = tf.get_default_graph()
    print("default_g:\n", default_g)

    # 方法2：查看属性
    print("a_t的图属性：\n", a_t.graph)
    print("c_t的图属性：\n", c_t.graph)

    # 自定义图
    new_g = tf.Graph()
    # 在自己的图中定义数据和操作
    with new_g.as_default():
        a_new = tf.constant(20)
        b_new = tf.constant(30)
        c_new = a_new + b_new
        print("a_new:\n", a_new)
        print("b_new:\n", b_new)
        print("c_new:\n", c_new)
        print("a_new的图属性：\n", a_new.graph)
        print("c_new的图属性：\n", c_new.graph)

    # 开启会话
    with tf.Session() as sess:
        # c_t_value = sess.run(c_t)
        print("c_t_value:\n", c_t.eval())
        print("sess的图属性:\n", sess.graph)
        # 1）将图写入本地生成events文件
        tf.summary.FileWriter("./tmp/summary", graph=sess.graph)

    # 开启new_g的会话
    with tf.Session(graph=new_g) as new_sess:
        c_new_value = new_sess.run(c_new)
        print("c_new_value:\n", c_new_value)
        print("new_sess的图属性：\n", new_sess.graph)
    return None


def session_demo():
    """
    会话的演示
    :return:
    """
    # TensorFlow实现加法运算
    a_t = tf.constant(2, name="a_t")
    b_t = tf.constant(3, name="b_t")
    c_t = tf.add(a_t, b_t, name="c_t")
    print("a_t:\n", a_t)
    print("b_t:\n", b_t)
    print("c_t：\n", c_t)

    # 定义占位符
    a_ph = tf.placeholder(tf.float32)
    b_ph = tf.placeholder(tf.float32)
    c_ph = tf.add(a_ph, b_ph)
    print("a_ph:\n", a_ph)
    print("b_ph:\n", b_ph)
    print("c_ph:\n", c_ph)

    # 查看默认图
    # 方法1：调用方法
    default_g = tf.get_default_graph()
    print("default_g:\n", default_g)

    # 方法2：查看属性
    print("a_t的图属性：\n", a_t.graph)
    print("c_t的图属性：\n", c_t.graph)

    # 开启会话
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=True)) as sess:
        # 运行placeholder
        c_ph_value = sess.run(c_ph, feed_dict={a_ph: 3.9, b_ph: 4.8})
        print("c_ph_value:\n", c_ph_value)
        # c_t_value = sess.run(c_t)
        # 同时查看a_t, b_t, c_t
        a, b, c = sess.run([a_t, b_t, c_t])
        print("abc:\n", a, b, c)
        print("c_t_value:\n", c_t.eval())
        print("sess的图属性:\n", sess.graph)
        # 1）将图写入本地生成events文件
        tf.summary.FileWriter("./tmp/summary", graph=sess.graph)

    return None


def tensor_demo():
    """
    张量的演示
    :return:
    """
    tensor1 = tf.constant(4.0)
    tensor2 = tf.constant([1, 2, 3, 4])
    linear_squares = tf.constant([[4], [9], [16], [25]], dtype=tf.int32)

    print("tensor1:\n", tensor1)
    print("tensor2:\n", tensor2)
    print("tensor_squares_before;\n", linear_squares)

    # 张量类型的修改
    l_cast = tf.cast(linear_squares, dtype=tf.float32)
    print("tensor_squares_last;\n", linear_squares)
    print("l_cast:\n", l_cast)

    # 更新/改变静态形状
    # 定义占位符
    # 没有完全固定下来的静态形状
    a_p = tf.placeholder(dtype=tf.float32, shape=[None, None])
    b_p = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    c_p = tf.placeholder(dtype=tf.float32, shape=[3, 2])
    print("a_p:\n", a_p)
    print("b_p:\n", b_p)
    print("c_p:\n", c_p)
    # 更新形状未确定的部分
    # a_p.set_shape([2, 3])
    # b_p.set_shape([2, 10])
    # c_p.set_shape([3, 2])
    # 动态形状修改
    a_p_reshape = tf.reshape(a_p, shape=[2, 3, 1])
    print("a_p:\n", a_p)
    # print("b_p:\n", b_p)
    print("a_p_shape:\n", a_p_reshape)
    c_p_reshape = tf.reshape(c_p, shape=[2, 3, 1])
    print("c_p:\n", c_p)
    print("c_p_reshape:\n", c_p_reshape)

    return None


def variable_demo():
    """
    变量的演示
    :return:
    """
    # 创建变量
    with tf.variable_scope("my_scope"):
        a = tf.Variable(initial_value=50)
        b = tf.Variable(initial_value=40)
    with tf.variable_scope("your_scoope"):
        c = tf.add(a, b)
    print("a:\n", a)
    print("b:\n", b)
    print("c:\n", c)

    # 初始化变量
    init = tf.global_variables_initializer()

    # 开启会话
    with tf.Session() as sess:
        # 运行初始化
        sess.run(init)
        a_value, b_value, c_value = sess.run([a, b, c])
        print("a_value:\n", a_value)
        print("b_value:\n", b_value)
        print("c_value:\n", c_value)

    return None


def linear_regression():
    """
    自实现一个线性回归
    :return:
    """
    with tf.variable_scope("prepare_data"):
        # 1）准备数据
        X = tf.random_normal(shape=[100, 1], name="feature")
        y_true = tf.matmul(X, [[0.8]]) + 0.7

    with tf.variable_scope("creat_model"):
        # 2）构造模型
        # 定义模型参数 用变量
        weights = tf.Variable(initial_value=tf.random_normal(shape=[1, 1]), name="Weights")
        bias = tf.Variable(initial_value=tf.random_normal(shape=[1, 1]), name="Bias")
        y_predict = tf.matmul(X, weights) + bias

    with tf.variable_scope("loss_function"):
        # 3）构造损失函数
        error = tf.reduce_mean(tf.square(y_predict - y_true))

    with tf.variable_scope("optimizer"):
        # 4）优化损失
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)

    # 2_收集变量
    tf.summary.scalar("error", error)
    tf.summary.histogram("weights", weights)
    tf.summary.histogram("bias", bias)

    # 3_合并变量
    merged = tf.summary.merge_all()

    # 创建Saver对象
    saver = tf.train.Saver()

    # 显示地初始化变量
    init = tf.global_variables_initializer()

    # 开启会话
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init)

        # 1_创建事件文件
        file_writer = tf.summary.FileWriter("./tmp/linear", graph=sess.graph)

        # 查看初始化模型参数之后的值
        print("训练前模型参数为：权重%f，偏置%f，损失为%f" % (weights.eval(), bias.eval(), error.eval()))

        # # 开始训练
        # for i in range(100):
        #     sess.run(optimizer)
        #     print("第%d次训练后模型参数为：权重%f，偏置%f，损失为%f" % (i+1, weights.eval(), bias.eval(), error.eval()))
        #
        #     # 运行合并变量操作
        #     summary = sess.run(merged)
        #     # 将每次迭代后的变量写入事件文件
        #     file_writer.add_summary(summary, i)
        #
        #     # 保存模型
        #     if i % 10 == 0:
        #         saver.save(sess, "./tmp/model/my_linear.ckpt")
        #  加载模型
        if os.path.exists("./tmp/model/checkpoint"):
            saver.restore(sess, "./tmp/model/my_linear.ckpt")

        print("训练后模型参数为：权重%f，偏置%f，损失为%f" % (weights.eval(), bias.eval(), error.eval()))

        return None


if __name__ == "__main__":
    # 代码1：TensorFlow的基本结构
    # tensorflow_demo()
    # 代码2：图的演示
    # graph_demo()
    # 代码3：会话的演示
    # session_demo()
    # 代码4：张量的演示
    # 代码5：变量的演示
    # variable_demo()
    # 代码6：自实现一个线性回归
    linear_regression()
