import tensorflow as tf

class Model:

    def __init__(self, _input_data, _learning_rate, _decay_rate, _batches_num, _global_step, _is_training):

        self.learning_rate = _learning_rate
        self.decay_rate = _decay_rate
        self.batches_num = _batches_num
        self.global_step = _global_step
        self.is_training = _is_training

        self.w_init = tf.initializers.he_normal()
        #self.w_init = tf.initializers.he_uniform()

        #self.w_init = tf.initializers.glorot_normal()
        #self.w_init = tf.initializers.glorot_uniform()

        #self.w_init = tf.initializers.lecun_normal()
        #self.w_init = tf.initializers.lecun_uniform()

        self._create_architecture(_input_data)

    def _create_architecture(self, _input_data):

        self.encoded = self._encoder(_input_data)
        self.embeddings = self._embedding(self.encoded)
        self.decoded = self._decoder(self.embeddings)

        self.loss = tf.losses.mean_squared_error(labels=_input_data, predictions=self.decoded)
        self.loss = tf.identity(self.loss, name='loss')
        self.rate_decay = tf.train.exponential_decay(self.learning_rate, self.global_step, self.batches_num, self.decay_rate, staircase=True, name='learning_rate_decay')

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.rate_decay, name='optimizer').minimize(loss=self.loss, global_step=self.global_step)
        #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.rate_decay, name='optimizer').minimize(loss=self.loss, global_step=self.global_step)
        #self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.rate_decay, name='optimizer').minimize(loss=self.loss, global_step=self.global_step)
        #self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.rate_decay, name='optimizer').minimize(loss=self.loss, global_step=self.global_step)
        #self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.rate_decay, name='optimizer').minimize(loss=self.loss, global_step=self.global_step)
        #self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.rate_decay, momentum=0.9, name='optimizer').minimize(loss=self.loss, global_step=self.global_step)
        #self.optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=self.rate_decay, name='optimizer').minimize(loss=self.loss, global_step=self.global_step)
        #self.optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate=self.rate_decay, name='optimizer').minimize(loss=self.loss, global_step=self.global_step)

    def _encoder(self, _input_data):

        # 28x28x1 -> 14x14x32 --------- 512x512x3 -> 256x256x32
        layer = tf.layers.conv2d(_input_data, filters=32, kernel_size=3, strides=2, padding='same', activation=tf.nn.leaky_relu, name='conv_1', kernel_initializer=self.w_init)
        layer = tf.layers.batch_normalization(layer, training=self.is_training, name='batch_norm_1')

        # 14x14x32 -> 7x7x64 --------- 256x256x32 -> 128x128x64
        layer = tf.layers.conv2d(layer, filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.leaky_relu, name='conv_2', kernel_initializer=self.w_init)
        layer = tf.layers.batch_normalization(layer, training=self.is_training, name='batch_norm_2')

        # 7x7x64 -> 3136 --------- 128x128x64 -> 1048576
        encoded = tf.layers.flatten(layer, name='flatten_1')

        return encoded

    def _embedding(self, _encoded):

        # 3136 -> 16 --------- 1048576 -> 16
        embeddings = tf.layers.dense(_encoded, 128, name='embeddings', kernel_initializer=self.w_init)

        return embeddings

    def _decoder(self, _embeddings):

        # 16 -> 3136 --------- 16 -> 1048576
        input = tf.layers.dense(_embeddings, 1048576, name='input_decoder', kernel_initializer=self.w_init)
        #input = tf.layers.dense(_embeddings, 3136, name='input_decoder', kernel_initializer=self.w_init)

        # 3136 -> 7x7x64 --------- 1048576 -> 128x128x64
        input = tf.reshape(input, shape=[-1, 128, 128, 64])
        #input = tf.reshape(input, shape=[-1, 7, 7, 64])

        # 7x7x64 -> 14x14x64 --------- 128x128x64 -> 256x256x64
        layer = tf.layers.conv2d_transpose(input, filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.leaky_relu, name='deconv_1', kernel_initializer=self.w_init)
        layer = tf.layers.batch_normalization(layer, training=self.is_training, name='batch_norm_deconv_1')

        # 14x14x64 -> 28x28x32 --------- 256x256x64 -> 512x512x32
        layer = tf.layers.conv2d_transpose(layer, filters=32, kernel_size=3, strides=2, padding='same', activation=tf.nn.leaky_relu, name='deconv_2', kernel_initializer=self.w_init)
        layer = tf.layers.batch_normalization(layer, training=self.is_training, name='batch_norm_deconv_2')

        # 512x512x32 -> 512x512x32
        decoded = tf.layers.conv2d_transpose(layer, filters=3, kernel_size=2, padding='same', activation=tf.nn.sigmoid, name='deconv_3', kernel_initializer=self.w_init)
        #decoded = tf.layers.conv2d_transpose(layer, filters=1, kernel_size=2, padding='same', activation=tf.nn.sigmoid, name='deconv_3', kernel_initializer=self.w_init)

        return decoded