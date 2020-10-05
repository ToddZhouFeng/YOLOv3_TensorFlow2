#修改为 yolo-fastest
#修改 ResidualBlock, 从原来的 ->1x1->3x3-> 变为 ->1x1->3x3->1x1->
#修改 make_residual_block, 增加前面的卷积层

import tensorflow as tf


class DarkNetConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, activation="leaky", groups=1):
        super(DarkNetConv2D, self).__init__()
        if groups != 1:
            self.conv = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size,
                                                        strides=strides,
                                                        padding="same")
        else:
            self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding="same")
        #self.conv = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME', data_format=format)
        self.bn = tf.keras.layers.BatchNormalization()
        if activation == "linear":
            self.activation = 1
        else:
            self.activation = 0.1

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=self.activation)
        return x


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters1, filters2):
        super(ResidualBlock, self).__init__()
        self.conv1 = DarkNetConv2D(filters=filters1, kernel_size=(1, 1), strides=1, activation="leaky")
        self.conv2 = DarkNetConv2D(filters=filters1, kernel_size=(3, 3), strides=1, activation="leaky", groups=filters1)
        self.conv3 = DarkNetConv2D(filters=filters2, kernel_size=(1, 1), strides=1, activation="linear")
        self.dropout1=tf.keras.layers.Dropout(0.15)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.dropout1(x, training=training)
        x = tf.keras.layers.add([x, inputs])
        return x


def make_residual_block(pre_conv_filters1, pre_conv_filters2, filters1,  filters2, num_blocks, pre_conv_strides=1):
    x = tf.keras.Sequential()
    x.add(DarkNetConv2D(filters=pre_conv_filters1, kernel_size=(3, 3), strides=pre_conv_strides, activation="leaky", groups=pre_conv_filters1))
    x.add(DarkNetConv2D(filters=pre_conv_filters2, kernel_size=(1, 1), strides=1, activation="linear"))

    for _ in range(num_blocks):
        x.add(ResidualBlock(filters1=filters1, filters2=filters2))
    return x


class YoloFastestBackbone(tf.keras.Model):
    def __init__(self):
        super(YoloFastestBackbone, self).__init__()
        self.conv1 = DarkNetConv2D(filters=8, kernel_size=(3, 3), strides=2, activation="leaky")
        self.conv2 = DarkNetConv2D(filters=8, kernel_size=(1, 1), strides=1, activation="leaky")
        self.block1 = make_residual_block(pre_conv_filters1=8, pre_conv_filters2=4, filters1=8,  filters2=4, num_blocks=1)
        
        self.conv3 = DarkNetConv2D(filters=24, kernel_size=(1, 1), strides=1, activation="leaky")
        self.block2 = make_residual_block(pre_conv_filters1=24, pre_conv_filters2=8, filters1=32,  filters2=8, num_blocks=2, pre_conv_strides=2)
        
        self.conv4 = DarkNetConv2D(filters=32, kernel_size=(1, 1), strides=1, activation="leaky")
        self.block3 = make_residual_block(pre_conv_filters1=32, pre_conv_filters2=8, filters1=48,  filters2=8, num_blocks=2, pre_conv_strides=2)

        self.conv5 = DarkNetConv2D(filters=48, kernel_size=(1, 1), strides=1, activation="leaky")
        self.block4 = make_residual_block(pre_conv_filters1=48, pre_conv_filters2=16, filters1=96,  filters2=16, num_blocks=4)

        self.conv6 = DarkNetConv2D(filters=96, kernel_size=(1, 1), strides=1, activation="leaky")
        self.block5 = make_residual_block(pre_conv_filters1=96, pre_conv_filters2=24, filters1=136,  filters2=24, num_blocks=4, pre_conv_strides=2)
        
        self.conv7 = DarkNetConv2D(filters=136, kernel_size=(1, 1), strides=1, activation="leaky")
        self.block6 = make_residual_block(pre_conv_filters1=136, pre_conv_filters2=48, filters1=224,  filters2=48, num_blocks=5, pre_conv_strides=2)



    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.block1(x, training=training)
        x = self.conv3(x, training=training)
        x = self.block2(x, training=training)
        x = self.conv4(x, training=training)
        x = self.block3(x, training=training)
        x = self.conv5(x, training=training)
        x = self.block4(x, training=training)
        x = self.conv6(x, training=training)
        x = self.block5(x, training=training)
        output_1 = self.conv7(x, training=training)
        output_2 = self.block6(output_1, training=training)
        # print(output_1.shape, output_2.shape, output_3.shape)
        return output_2, output_1


class YOLOTail(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super(YOLOTail, self).__init__()
        self.conv1 = DarkNetConv2D(filters=in_channels, kernel_size=(1, 1), strides=1, activation="leaky")
        self.conv2 = DarkNetConv2D(filters=in_channels, kernel_size=(3, 3), strides=1, activation="leaky", groups=in_channels)
        self.conv3 = DarkNetConv2D(filters=in_channels, kernel_size=(3, 3), strides=1, activation="leaky", groups=in_channels)
        self.conv4 = DarkNetConv2D(filters=4*in_channels/3, kernel_size=(1, 1), strides=1, activation="linear")
        self.conv5 = DarkNetConv2D(filters=4*in_channels/3, kernel_size=(3, 3), strides=1, activation="leaky", groups=int(4*in_channels/3))
        self.conv6 = DarkNetConv2D(filters=4*in_channels/3, kernel_size=(3, 3), strides=1, activation="leaky", groups=int(4*in_channels/3))
        self.conv7 = DarkNetConv2D(filters=4*in_channels/3, kernel_size=(1, 1), strides=1, activation="linear")
        
        self.normal_conv = tf.keras.layers.Conv2D(filters=out_channels,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding="same")

    def call(self, inputs, training=None, **kwargs):
        branch = self.conv1(inputs, training=training)
        x = self.conv2(branch, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        x = self.conv6(x, training=training)
        x = self.conv7(x, training=training)
        stem = self.normal_conv(x)
        return stem, branch


class YOLOV3(tf.keras.Model):
    def __init__(self, out_channels):
        super(YOLOV3, self).__init__()
        self.darknet = YoloFastestBackbone()
        self.tail_1 = YOLOTail(in_channels=96, out_channels=out_channels)
        self.upsampling_1 = self._make_upsampling(num_filter=256)
        self.tail_2 = YOLOTail(in_channels=96, out_channels=out_channels)
        #self.upsampling_2 = self._make_upsampling(num_filter=256)
        #self.tail_3 = YOLOTail(in_channels=256, out_channels=out_channels)

    def _make_upsampling(self, num_filter):
        layer = tf.keras.Sequential()
        #layer.add(DarkNetConv2D(filters=num_filter, kernel_size=(1, 1), strides=1))
        layer.add(tf.keras.layers.UpSampling2D(size=(2, 2)))
        return layer

    def call(self, inputs, training=None, mask=None):
        x_1, x_2 = self.darknet(inputs, training=training)
        stem_1, branch_1 = self.tail_1(x_1, training=training)
        branch_1 = self.upsampling_1(branch_1, training=training)
        x_2 = tf.keras.layers.concatenate([branch_1, x_2])
        stem_2, _ = self.tail_2(x_2, training=training)
        #branch_2 = self.upsampling_2(branch_2, training=training)
        #x_3 = tf.keras.layers.concatenate([branch_1, x_3])
        #stem_3, _ = self.tail_3(x_3, training=training)

        return [stem_1, stem_2]

