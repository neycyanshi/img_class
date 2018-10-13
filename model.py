import tensorflow as tf


def base_drop(inputs, mode, scope='base_drop'):
    """Return the output operation following the network architecture.
    Args:
        inputs (Tensor): Input Tensor
        mode (ModeKeys): Runtime mode (train, eval, predict)
        scope (str): Name of the scope of the architecture
    Returns:
         Logits output Op for the network.
    """
    with tf.variable_scope(scope):
        # inputs = inputs / 255
        inputs = tf.reshape(inputs, [-1, 28, 28, 1])
        conv1 = tf.layers.conv2d(
            inputs=inputs,
            filters=20,
            kernel_size=[5, 5],
            padding='valid',
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=40,
            kernel_size=[5, 5],
            padding='valid',
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2, pool_size=[2, 2], strides=2)
        flatten = tf.reshape(pool2, [-1, 4 * 4 * 40])
        dense1 = tf.layers.dense(
            inputs=flatten, units=256, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense1,
            rate=0.5,
            training=mode == tf.estimator.ModeKeys.TRAIN)
        dense2 = tf.layers.dense(inputs=dropout, units=10)
        return dense2


def base(inputs, training, scope='base'):
    """
    Create a (very simple) TF model to classify MNIST digits.

    :param inputs: a Tensor with the image inputs (N, 32, 32, 1)
    :param training: a bool or Tensor(tf.bool) indicating training phase
    :return: the logits (pre-softmax) output Tensor
    """
    with tf.variable_scope(scope):
        net = tf.layers.conv2d(
            inputs, 32, 5, padding='same', activation=tf.nn.relu, name='conv1')
        net = tf.layers.max_pooling2d(net, 2, 2, name='pool1')
        net = tf.layers.conv2d(
            net, 64, 5, padding='same', activation=tf.nn.relu, name='conv2')
        net = tf.layers.max_pooling2d(net, 2, 2, name='pool2')
        net = tf.layers.flatten(net, name='flatten')
        net = tf.layers.dense(net, 1024, activation=tf.nn.relu, name='fc3')
        net = tf.layers.dropout(net, training)
        net = tf.layers.dense(net, 10, name='logits')
        return net


class WideResNet:
    # pRN & wRN
    # wRN is just (x4) increased filter pRN with decreased depth + dropout
    # in this case, we have to consider first block (no need BN & ReLU)
    def __init__(self, mode, dropout_rate=0.3, layer_n=2, name='wide_resnet'):
        self.training = (mode == tf.estimator.ModeKeys.TRAIN)
        self.dropout_rate = dropout_rate
        self.layer_n = 2
        self.seed = 777

    def wide_residual_block(self,
                            x,
                            output_channel,
                            first_block=False,
                            downsampling=False,
                            name='wRN_block'):
        if downsampling:
            stride = 2
        else:
            stride = 1

        net = x

        with tf.variable_scope(name):
            with tf.variable_scope('conv1_in_block'):
                if not first_block:
                    net = tf.layers.batch_normalization(
                        net, training=self.training)
                    net = tf.nn.relu(net)
                net = tf.layers.conv2d(
                    net,
                    output_channel, [3, 3],
                    strides=stride,
                    padding='SAME',
                    use_bias=False,
                    kernel_initializer=tf.contrib.layers.
                    variance_scaling_initializer(seed=self.seed))

            if self.dropout_rate > 0.0:
                net = tf.layers.dropout(
                    net,
                    rate=self.dropout_rate,
                    training=self.training,
                    seed=self.seed,
                    name='dropout_in_block')

            with tf.variable_scope('conv2_in_block'):
                net = tf.layers.batch_normalization(
                    net, training=self.training)
                net = tf.nn.relu(net)
                net = tf.layers.conv2d(
                    net,
                    output_channel, [3, 3],
                    strides=1,
                    padding='SAME',
                    use_bias=False,
                    kernel_initializer=tf.contrib.layers.
                    variance_scaling_initializer(seed=self.seed))

            if downsampling:
                x = tf.layers.conv2d(
                    x,
                    output_channel, [1, 1],
                    strides=stride,
                    padding='SAME',
                    kernel_initializer=tf.contrib.layers.
                    variance_scaling_initializer(seed=self.seed))
            # TODO: check wide resnet for first layer dimension matching
            # elif first_block: # first_block is not downsampling
            #     x = tf.layers.conv2d(x, output_channel, [1,1], strides=1, padding='SAME',
            #         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(seed=self.seed))

            return net + x

    def forward(self, x_img):
        net = tf.reshape(x_img, [-1, 28, 28, 1])

        # conv0: conv-bn-relu [-1, 28, 28, 16]
        # conv1: [-1, 28, 28, 64] * n
        # conv2: [-1, 14, 14, 128] * n
        # conv3: [-1,  7,  7, 256] * n
        # global average pooling
        # dense
        # widening factor = 4

        with tf.variable_scope("conv0"):
            net = tf.layers.conv2d(
                net,
                64, [3, 3],
                strides=1,
                padding='SAME',
                use_bias=False,
                kernel_initializer=tf.contrib.layers.
                variance_scaling_initializer(seed=self.seed))
            net = tf.layers.batch_normalization(net, training=self.training)
            net = tf.nn.relu(net)

        with tf.variable_scope("conv1"):
            for i in range(self.layer_n):
                net = self.wide_residual_block(
                    net,
                    64,
                    first_block=(i == 0),
                    name="resblock{}".format(i + 1))
                assert net.shape[1:] == [28, 28, 64]

        with tf.variable_scope("conv2"):
            for i in range(self.layer_n):
                net = self.wide_residual_block(
                    net,
                    128,
                    downsampling=(i == 0),
                    name="resblock{}".format(i + 1))
                assert net.shape[1:] == [14, 14, 128]

        with tf.variable_scope("conv3"):
            for i in range(self.layer_n):
                net = self.wide_residual_block(
                    net,
                    256,
                    downsampling=(i == 0),
                    name="resblock{}".format(i + 1))
                assert net.shape[1:] == [7, 7, 256]

        with tf.variable_scope("fc"):
            net = tf.layers.batch_normalization(net, training=self.training)
            net = tf.nn.relu(net)
            net = tf.reduce_mean(net, [1, 2])  # global average pooling
            assert net.shape[1:] == [256]

            logits = tf.layers.dense(
                net,
                10,
                kernel_initializer=tf.contrib.layers.
                variance_scaling_initializer(seed=self.seed),
                name="logits")

        return logits
