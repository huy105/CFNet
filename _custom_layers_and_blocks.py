import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

################################################################################
# Layers
################################################################################
class ConvolutionBnActivation(tf.keras.layers.Layer):
    """
    """
    # def __init__(self, filters, kernel_size, strides=(1, 1), activation=tf.keras.activations.relu, **kwargs):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding="same", data_format=None, dilation_rate=(1, 1),
                 groups=1, activation=None, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, use_batchnorm=False, 
                 axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, trainable=True,
                 post_activation="relu", block_name=None):
        super(ConvolutionBnActivation, self).__init__()


        # 2D Convolution Arguments
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.use_bias = not (use_batchnorm)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        # Batch Normalization Arguments
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.trainable = trainable
        
        self.block_name = block_name
        
        self.conv = None
        self.bn = None
        #tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)
        self.post_activation = tf.keras.layers.Activation(post_activation)


    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name=self.block_name + "_conv" if self.block_name is not None else None

        )

        self.bn = tf.keras.layers.BatchNormalization(
            axis=self.axis,
            momentum=self.momentum,
            epsilon=self.epsilon,
            center=self.center,
            scale=self.scale,
            trainable=self.trainable,
            name=self.block_name + "_bn" if self.block_name is not None else None
        )

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.post_activation(x)

        return x

    def compute_output_shape(self, input_shape):
        print(input_shape)
        return [input_shape[0], input_shape[1], input_shape[2], self.filters]


class MixtureOfSoftMaxACF(tf.keras.layers.Layer):

    """
        self.temperature: Scale score in attention https://www.youtube.com/watch?v=4Bdc55j80l8&ab_channel=TheA.I.Hacker-MichaelPhi
        self.att_dropout: Drop out layer
        n_mix: number of mixtures of softmax
        d_k: number of filter
    
    """

    def __init__(self, n_mix = 1, d_k = 0, att_dropout=0.1):
        super(MixtureOfSoftMaxACF, self).__init__()
        self.temperature = tf.math.pow(tf.cast(d_k, tf.float32), 0.5)  # 
        self.att_dropout = att_dropout
        self.n_mix = n_mix
        self.dropout = tf.keras.layers.Dropout(att_dropout)
        self.softmax_1 = tf.keras.layers.Softmax(axis = 1) 
        self.softmax_2 = tf.keras.layers.Softmax(axis = 2)
        self.d_k = d_k

        #  self.kernel (self.n_mix, self.d_k)
        if self.n_mix > 1:
            std = np.power(n_mix, -0.5)
            init_weight = tf.keras.initializers.RandomUniform(minval = -std, maxval = std, seed=None)
            self.kernel = self.add_weight(name='weight_k',
                                      shape=(self.n_mix, self.d_k),
                                      initializer = init_weight,
                                      trainable=True)


    def call(self, qt, kt, vt, training=None):
        m = self.n_mix

        if K.image_data_format() == "channels_last":
            BS, N, d_k = qt.shape # (BS, H * W, C)

            assert d_k == self.d_k  # check input with qt dimension size
            d = d_k // m     # divide filter by number of mixture

            if m > 1:
                bar_qt = tf.math.reduce_mean(qt, axis=1, keepdims=False)  # (BS, d_k)
                prior_mix = self.softmax_1(tf.linalg.matmul(self.kernel, bar_qt, transpose_b=True)) # matmul (BS, n_mix)
                prior_mix = tf.reshape(prior_mix, (BS * m, 1, 1))

            q = tf.reshape(qt, (BS * m, N, d))                      # (BS * m, N, d)
            # q = tf.transpose(q, perm=[0, 2, 1])                   # (BS, d, N)
            N2 = kt.shape[1]
            kt = tf.reshape(kt, (BS * m, N2, d))                    # (BS * m, N2, d)
            # v = tf.keras.layers.transpose(vt, perm=[0, 2, 1])     # (BS, d, N2)

            att = tf.linalg.matmul(q, kt, transpose_b=True)         # (BS * m, N, N2)
            att = att / self.temperature                            # (BS * m, N, N2)
            att = self.softmax_2(att)                               # (BS * m, N, N2)
            att = self.dropout(att, training=training)              # (BS * m, N, N2)

            if m > 1:
                att = (att * prior_mix)
                att = tf.reshape(att, (BS, m, N, N2))
                att = tf.math.reduce_sum(att, axis=1, keepdims=False)  

            out = tf.linalg.matmul(att, vt)                     # (BS, N, d)

        else:
            BS, d_k, N = qt.shape

            assert d_k == self.d_k
            d = d_k // m
            
            if m > 1:
                bar_qt = tf.math.reduce_mean(qt, axis=2, keepdims=False)  # (BS, d_k, 1)
                prior_mix = self.softmax_1(tf.linalg.matmul(self.kernel, bar_qt)) # matmul (BS, n_mix)
                prior_mix = tf.reshape(prior_mix, (BS * m, 1, 1))


            q = tf.reshape(qt, (BS * m, d, N))(qt)              # (BS * m, d, N)
            # q = tf.transpose(q, perm=[0, 2, 1])               # (BS, N, d)
            N2 = kt.shape[2]
            kt = tf.reshape(kt, (BS * m, d, N2))                # (BS * m, d, N2)
            # v = tf.transpose(vt, perm=[0, 2, 1])              # (BS * m, N2, d)

            att = tf.linalg.matmul(q, kt, transpose_a=True)     # (BS * m, N, N2)
            att = att / self.temperature                        # (BS * m, N, N2)
            att = self.softmax_2(att)                           # (BS * m, N, N2)
            att = self.dropout(att, training=training)          # (BS * m, N, N2)

            if m > 1:
                att = (att * prior_mix)
                att = tf.reshape(att, (BS, m, N, N2))
                att = tf.math.reduce_sum(att, axis=1, keepdims=False)  

            out = tf.linalg.matmul(att, vt, transpose_b=True)   # (BS, N, d)

        return out


class GlobalPooling(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(GlobalPooling, self).__init__()
        self.filters = filters

        self.conv1x1_bn_relu = ConvolutionBnActivation(filters, (1, 1))
    

    def call(self, x, training=None):
        if K.image_data_format() == "channels_last":
            BS, H, W, C = x.shape
            glob_avg_pool = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2], keepdims=True))(x)
            glob_avg_pool = self.conv1x1_bn_relu(glob_avg_pool, training=training)
            glob_avg_pool = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, [H, W]))(glob_avg_pool)
        else:
            BS, C, H, W = x.shape
            glob_avg_pool = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2], keepdims=True))(x)
            glob_avg_pool = self.conv1x1_bn_relu(glob_avg_pool, training=training)
            glob_avg_pool = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, [H, W]))(glob_avg_pool)

        return glob_avg_pool


