import tensorflow as tf
import tensorflow.keras.backend as K
from _custom_layers_and_blocks import ConvolutionBnActivation, GlobalPooling
from AggCF_Module import AggCF_Module


class CFNet(tf.keras.Model):
    # Co-occurent Feature Network
    def __init__(self, n_classes, base_model, output_layers, height=None, width=None, filters=512, 
                n_heads=8, n_mix = 1, final_activation="softmax", backbone_trainable=False,
                lateral=True, global_pool=False, acf_pool=True,
                acf_kq_transform="ffn", acf_concat=False, **kwargs):
        super(CFNet, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.backbone = None
        self.filters = filters
        self.final_activation = final_activation
        self.lateral = lateral
        self.global_pool = global_pool
        self.acf_pool = acf_pool
        self.acf_kq_transform = acf_kq_transform
        self.acf_concat = acf_concat
        self.height = height
        self.width = width
        self.n_heads = n_heads 
        self.n_mix = n_mix 


        output_layers = output_layers[1:5]

        base_model.trainable = backbone_trainable
        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=output_layers)

        # Layers
        self.conv3x3_bn_relu_1 = ConvolutionBnActivation(filters, (3, 3))
        self.conv1x1_bn_relu_2 = ConvolutionBnActivation(filters, (1, 1))
        self.conv1x1_bn_relu_3 = ConvolutionBnActivation(filters, (1, 1))
        self.conv3x3_bn_relu_4 = ConvolutionBnActivation(filters, (3, 3))

        self.upsample2d_2x = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")
        self.upsample2d_4x = tf.keras.layers.UpSampling2D(size=4, interpolation="bilinear")
        self.pool2d = tf.keras.layers.MaxPooling2D((2, 2), padding="same")

        axis = 3 if K.image_data_format() == "channels_last" else 1
        self.concat_1 = tf.keras.layers.Concatenate(axis=axis)
        self.concat_2 = tf.keras.layers.Concatenate(axis=axis)

        self.glob_pool = GlobalPooling(filters)

        d_k = filters // self.n_heads * n_mix
        d_v = filters // self.n_heads
        self.acf = AggCF_Module(filters, d_k = d_k, d_v = d_v, n_heads = self.n_heads, n_mix = self.n_mix ,
                            kq_transform=self.acf_kq_transform, value_transform="conv",
                            pooling=self.acf_pool, concat=self.acf_concat, dropout=0.1)
        
        self.final_conv3x3_bn_activation = ConvolutionBnActivation(n_classes, (3, 3), post_activation=final_activation)
        self.final_upsampling2d = tf.keras.layers.UpSampling2D(size=8, interpolation="bilinear")

    def call(self, inputs, training=None, mask=None):
        if training is None or training is False:
            training = True

        assert training == True
        # x3: 7x7, 2048
        # x2: 14x14, 1024
        # x1: 28x28, 512
        x0, x1, x2, x3 = self.backbone(inputs, training=training)

        feat = self.conv3x3_bn_relu_1(x3, training=training)
        feat = self.upsample2d_4x(feat)
        if self.lateral:
            c3 = self.conv1x1_bn_relu_3(x2, training=training)
            c3 = self.upsample2d_2x(c3)
            c2 = self.conv1x1_bn_relu_2(x1, training=training)
            
            feat = self.concat_1([feat, c2, c3])
            feat = self.conv3x3_bn_relu_4(feat, training=training)

        if self.global_pool:
            pool = self.glob_pool(feat, training=training)
            feat = self.acf(feat, training=training)
            feat = self.concat_2([pool, feat])
        else:
            feat = self.acf(feat, training=training)

        x = self.final_conv3x3_bn_activation(feat, training=training)
        x = self.final_upsampling2d(x)

        return x

    def model(self):
        x = tf.keras.layers.Input(shape=(self.height, self.width, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
