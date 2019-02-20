import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorflow.contrib import layers


class Rnn_:
    def __init__(self, name, data):
        self.name = name
        self.data = data

    def get_pre(self, data9, NUM_STEPS):
        # 最后一行不应该参与标准差和均值计算
        max = np.max(data9, axis=0)
        min = np.min(data9, axis=0)
        # normalized_train_data = (data9 - min) / (max - min)
        normalized_train_data = data9
        # 使用所有样本进行数据的标准化处理，取出最后一条数据进行预测
        train_x = []
        size = len(normalized_train_data) // NUM_STEPS
        for i in range(size):
            x = normalized_train_data[i * NUM_STEPS: (i + 1) * NUM_STEPS, : -1]
            train_x.append(x)
        return np.array(train_x), max, min

    def get_train(self, data_train, NUM_STEPS):
        self.NUM_STEPS = NUM_STEPS
        train_x, train_y = [], []
        # normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # 标准化
        max = np.max(data_train, axis=0)
        min = np.min(data_train, axis=0)
        # normalized_train_data = (data_train - min) / (max - min)
        normalized_train_data = data_train
        # label不进行归一化
        for i in range(len(normalized_train_data) - NUM_STEPS):
            # x = normalized_train_data[i: i + NUM_STEPS, : -1]
            x = data_train[i: i + NUM_STEPS, : -1]
            # y = normalized_train_data[i: i + NUM_STEPS, -1, None]
            y = data_train[i: i + NUM_STEPS, -1, None]
            train_x.append(x)
            train_y.append(y)

        return np.array(train_x), np.array(train_y)

    def rnn_network(self, x):
        network = tl.layers.InputLayer(x, 'input_layer_new')
        network = tl.layers.Conv1dLayer(network, name='conv1')
        network = tl.layers.DynamicRNNLayer(network, tf.nn.rnn_cell.LSTMCell, n_hidden=72, return_last=False,
                                            return_seq_2d=True, name='rnn11')
        network = tl.layers.DropoutLayer(network)
        network = tl.layers.DenseLayer(network, 1, name='output_layer1')
        return network

    # 自带标准化
    # num_step在这里以为着句长/检测时间长
    def train(self, num_step, batch_size, data, model_dir, input_size):
        output_size = 1
        train_x, train_y = self.get_train(data, num_step)
        x = tf.placeholder(tf.float32, shape=[None, num_step, input_size])
        y = tf.placeholder(tf.float32, shape=[None, num_step, output_size])
        rnn_network = self.rnn_network(x)
        y_o = rnn_network.outputs
        y_ = tf.reshape(y_o, [-1, num_step, output_size])
        cost = tf.reduce_mean(tf.square(y_ - y))
        correct_prediction = tf.equal(y, y_)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        train_params = rnn_network.all_params
        lr = tf.Variable(0.001, trainable=False)
        train_op = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-08,
                                          use_locking=False).minimize(cost, var_list=train_params)
        # train_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(cost, var_list=train_params)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33333)
        config = tf.ConfigProto(
            gpu_options=gpu_options,
            device_count={'GPU': 1},
            log_device_placement=True,
            allow_soft_placement=True,
        )
        config.gpu_options.allow_growth = True
        sess = tf.InteractiveSession(config=config)
        tl.layers.initialize_global_variables(sess)
        tl.files.load_and_assign_npz(network=rnn_network, name=model_dir, sess=sess)
        rnn_network.print_params()
        rnn_network.print_layers()
        for i in range(100):
            sess.run(tf.assign(lr, 0.002 * (0.97 ** i)))
            tl.utils.fit(sess, rnn_network, train_op, cost, train_x, train_y, x, y, acc=acc, batch_size=batch_size,
                         n_epoch=50, print_freq=1, eval_train=False)
            if i % 10 == 0:
                tl.files.save_npz(rnn_network.all_params, name=model_dir)
            print("epoch: %d" % i)
        tl.files.save_npz(rnn_network.all_params, name=model_dir)


    def predict(self, data, model_dir, NUM_STEPS):
        predict_x, max, min = self.get_pre(data, NUM_STEPS)
        x = tf.placeholder(tf.float32, shape=[None, predict_x.shape[1], predict_x.shape[2]])
        rnn_network = self.rnn_network(x)
        y_o = rnn_network.outputs
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33333)
        config = tf.ConfigProto(
            gpu_options=gpu_options,
            device_count={'GPU': 0},
            log_device_placement=True,
            allow_soft_placement=True,
        )
        with tf.Session(config=config) as sess:
            tl.layers.initialize_global_variables(sess)
            rnn_network.print_params()
            rnn_network.print_layers()
            tl.files.load_and_assign_npz(network=rnn_network, name=model_dir, sess=sess)
            result = []
            for i in range(predict_x.shape[0]):
                # 变成3维数据, 投入网络预测
                X = predict_x[i, :, :]
                X = np.reshape(X, [-1, X.shape[0], X.shape[1]])
                prediction = tl.utils.predict(sess, rnn_network, X, x, y_o, 1)

                result.extend(prediction)
                pass
            result = np.array(result)
            return result

