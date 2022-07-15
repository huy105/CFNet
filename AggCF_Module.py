import tensorflow as tf
import tensorflow.keras.backend as K
from _custom_layers_and_blocks import MixtureOfSoftMaxACF, ConvolutionBnActivation



class AggCF_Module(tf.keras.layers.Layer):

    """
        Build function: For define the feet forward network (3 options)
        n_mix: Number of mixtures of softmax
        n_head: Number of head using by multi attention
        filters: Number of dimesion of query
        d_k: Number of dimesion of key (filters// n_head * n_mix)
        d_v: Number of dimesion of value (filters // n_head)
    """

    def __init__(self, filters = 512, d_k = 256, d_v = 256, n_heads = 8, n_mix = 1, 
                    kq_transform="ffn", value_transform="ffn", pooling=True, concat=False, dropout=0.1):
        super(AggCF_Module, self).__init__()
        self.filters = filters
        self.kq_transform = kq_transform
        self.value_transform = value_transform
        self.pooling = pooling
        self.concat = concat # if True concat else Add
        self.dropout = dropout
        self.n_mix = n_mix
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v


        # original from owner
        # self.avg_pool2d_1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding="same", data_format=K.image_data_format())
        # self.avg_pool2d_2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding="same", data_format=K.image_data_format())

        # implement like in paper
        self.avg_pool2d_1 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides = 2, padding="same", data_format=K.image_data_format())
        self.avg_pool2d_2 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides = 2, padding="same", data_format=K.image_data_format())
        
        self.conv_ks_1 = None       #feed forward (key) network
        self.conv_ks_2 = None       #feed forward (key) network
        self.conv_vs = None         #feed forward (value) network

        self.attention = MixtureOfSoftMaxACF(n_mix = n_mix, d_k = self.d_k, att_dropout=0.1)

        self.conv1x1_bn_relu = tf.keras.layers.Conv2D(filters, (1, 1), padding="same")
        
        axis = 3 if K.image_data_format() == "channels_last" else 1
        self.bn = tf.keras.layers.BatchNormalization(axis=axis)
        self.concat_mtr = tf.keras.layers.Concatenate(axis=axis)
        self.add = tf.keras.layers.Add()

        self.init_weight1 = tf.keras.initializers.RandomNormal(mean=0, stddev=tf.math.sqrt(2 / (self.filters + self.d_k)))
        self.init_weight2 = tf.keras.initializers.RandomNormal(mean=0, stddev=tf.math.sqrt(1 / self.d_k))
        self.init_weight3 = tf.keras.initializers.RandomNormal(mean=0, stddev=tf.math.sqrt(2 / (self.filters + self.d_v)))

    def build(self, input_shape):
        if self.kq_transform == "conv":
            self.conv_ks_1 = tf.keras.layers.Conv2D(self.n_heads * self.d_k, (1, 1), padding="same", kernel_initializer = self.init_weight1)
            self.conv_ks_2 = tf.keras.layers.Conv2D(self.n_heads * self.d_k, (1, 1), padding="same", kernel_initializer = self.init_weight1)
        elif self.kq_transform == "ffn":
            self.conv_ks_1 = tf.keras.Sequential(
                [ConvolutionBnActivation(self.n_heads * self.d_k, (3, 3), kernel_initializer = self.init_weight2),
                tf.keras.layers.Conv2D(self.n_heads * self.d_k, (1, 1), padding="same", kernel_initializer = self.init_weight2)]
            )
            self.conv_ks_2 = tf.keras.Sequential(
                [ConvolutionBnActivation(self.n_heads * self.d_k, (3, 3), kernel_initializer = self.init_weight2),
                tf.keras.layers.Conv2D(self.n_heads * self.d_k, (1, 1), padding="same", kernel_initializer = self.init_weight2)]
            )
        elif self.kq_transform == "dffn":
            self.conv_ks_1 = tf.keras.Sequential(
                [ConvolutionBnActivation(self.n_heads * self.d_k, (3, 3), dilation_rate=(4, 4), kernel_initializer = self.init_weight2),
                tf.keras.layers.Conv2D(self.n_heads * self.d_k, (1, 1), padding="same", kernel_initializer = self.init_weight2)]
            )
            self.conv_ks_2 = tf.keras.Sequential(
                [ConvolutionBnActivation(self.n_heads * self.d_k, (3, 3), dilation_rate=(4, 4), kernel_initializer = self.init_weight2),
                tf.keras.layers.Conv2D(self.n_heads * self.d_k, (1, 1), padding="same", kernel_initializer = self.init_weight2)]
            )
        else:
            raise NotImplementedError("Allowed options for 'kq_transform' are only ('conv', 'ffn', 'dffn'), got {}".format(self.kq_transform))
        
        if self.value_transform == "conv":
            self.conv_vs = tf.keras.layers.Conv2D(self.n_heads * self.d_v, (1, 1), padding="same", kernel_initializer = self.init_weight3)
        else:
            raise NotImplementedError("Allowed options for 'value_transform' is only 'conv', got {}".format(self.kq_transform))

    def call(self, x, training=None):
        residual = x
        d_k, d_v, n_heads = self.d_k, self.d_v, self.n_heads

        # After avgpooling size image reduce by 2, so we will using H * W // 4 to back to original size
        if K.image_data_format() == "channels_last":
            BS, H, W, C = x.shape                                                   #x(8, 28, 28, 512)
            
            if self.pooling:
                qt = self.conv_ks_1(x, training=training)                           # (BS, N, C: n_heads * d_k)
                qt = tf.reshape(qt, (BS * n_heads, H * W, -1))                      # (BS * n_heads, N, C: d_k
                kt = self.avg_pool2d_1(x)
                kt = self.conv_ks_2(kt, training=training)
                kt = tf.reshape(kt, (BS * n_heads, H * W // 4, -1))                 # (BS * n_heads, N / 4, C: d_k)
                vt = self.avg_pool2d_2(x)
                vt = self.conv_vs(vt, training=training)
                vt = tf.reshape(vt, (BS * n_heads, H * W // 4, -1))                 # (BS * n_heads , N / 4, C: d_v)
            else:
                qt = self.conv_ks_1(x, training=training)
                qt = tf.reshape(qt, (BS * n_heads, H * W, -1))                      # (BS * n_heads, N, C: d_k)
                kt = self.conv_ks_2(x, training=training)
                kt = tf.reshape(kt, (BS * n_heads, H * W // 4, -1))                 # (BS * n_heads, N / 4, C: d_k)
                vt = self.conv_vs(x, training=training)
                vt = tf.reshape(vt, (BS * n_heads, H * W // 4, -1))                 # (BS * n_heads , N / 4, C: d_v)

            out = self.attention(qt, kt, vt, training=training)                     # (BS * n_heads, N, C)

            # out = tf.transpose(out, perm=[0, 2, 1])                 
            out = tf.reshape(out, (BS, H, W, -1))                                   # (BS, H, W, C: d_v * n_heads) 
            
        else:
            BS, C, H, W = x.shape

            if self.pooling:
                qt = self.conv_ks_1(x, training=training)
                qt = tf.reshape(qt, (BS * n_heads, -1, H * W))                      # (BS * n_heads, d_k, N)
                kt = self.avg_pool2d_1(x)
                kt = self.conv_ks_2(kt, training=training)
                kt = tf.reshape(kt, (BS * n_heads, -1, H * W // 4))                 # (BS * n_heads, d_k, N / 4)
                vt = self.avg_pool2d_2(x)
                vt = self.conv_vs(vt, training=training)
                vt = tf.reshape(vt, (BS * n_heads, -1, H * W // 4))                 # (BS * n_heads, d_v, N / 4)
            else:
                qt = self.conv_ks_1(x, training=training)
                qt = tf.reshape(qt, (BS * n_heads, -1, H * W))                      # (BS * n_heads, d_k, N)
                kt = self.conv_ks_2(x, training=training)
                kt = tf.reshape(kt, (BS * n_heads, -1, H * W))                      # (BS * n_heads, d_k, N)
                vt = self.conv_vs(x, training=training)
                vt = tf.reshape(vt, (BS * n_heads, -1, H * W))                      # (BS * n_heads, d_v, N)

            out = self.attention(qt, kt, vt)                        # (BS * n_heads, N, C)

            out = tf.transpose(out, perm=[0, 2, 1])                 # (BS * n_heads, C, N)
            out = tf.reshape(out, (BS, -1, H, W))                   # (BS, C: d_v * n_heads, H, W)

        out = self.conv1x1_bn_relu(out, training=training)
        if self.concat:
            out = self.concat_mtr([out, residual])
        else:
            out = self.add([out, residual])
              
        return out

