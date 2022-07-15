import tensorflow as tf

################################################################################
# Backbone
################################################################################
def create_base_model(name="ResNet50", weights="imagenet", height=None, width=None,
                      channels=3, include_top=False, pooling=None, alpha=1.0,
                      depth_multiplier=1, dropout=0.001):
    if not isinstance(height, int) or not isinstance(width, int) or not isinstance(channels, int):
        raise TypeError("'height', 'width' and 'channels' need to be of type 'int'")
        
    if channels <= 0:
        raise ValueError(f"'channels' must be greater of equal to 1 but given was {channels}")
    
    input_shape = [height, width, channels]

    if name.lower() == "resnet101":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.ResNet101(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
        layer_names = ["conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block23_out", "conv5_block3_out"]

    layers = [base_model.get_layer(layer_name).output for layer_name in layer_names]
    
    return base_model, layers, layer_names
