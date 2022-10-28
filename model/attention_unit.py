from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    LeakyReLU,
    Permute,
    Dot,
    Softmax,
    Reshape,
    Add,
)
from tensorflow.keras.regularizers import l2
import tensorflow as tf


class Attention(tf.keras.layers.Layer):
    def __init__(self, name="attention_block", **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shapes):
        self.alpha = self.add_weight(
            shape=(), initializer=tf.initializers.Zeros, trainable=True
        )

    def call(self, inputs):
        return tf.multiply(self.alpha, inputs)


def attention_block(
    inputs,
    filters=64,
    leaky_value=0,
    initializer="glorot_uniform",
    l2_weight=0.05,
    model_name="attention_block",
):
    # position attention module
    conv_s = Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        kernel_initializer=initializer,
        kernel_regularizer=l2(l2_weight),
        padding="same",
        name="{}_conv_s".format(model_name),
    )(inputs)
    conv_s = BatchNormalization(name="{}_bn_s".format(model_name))(conv_s)
    conv_s = LeakyReLU(alpha=leaky_value, name="{}_act_s".format(model_name))(conv_s)

    conv_s1 = Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        kernel_initializer=initializer,
        kernel_regularizer=l2(l2_weight),
        padding="same",
        name="{}_conv_s1".format(model_name),
    )(conv_s)
    conv_s1 = BatchNormalization(name="{}_bn_s1".format(model_name))(conv_s1)
    conv_s1 = LeakyReLU(alpha=leaky_value, name="{}_act_s1".format(model_name))(conv_s1)

    conv_s2 = Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        kernel_initializer=initializer,
        kernel_regularizer=l2(l2_weight),
        padding="same",
        name="{}_conv_s2".format(model_name),
    )(conv_s)
    conv_s2 = BatchNormalization(name="{}_bn_s2".format(model_name))(conv_s2)
    conv_s2 = LeakyReLU(alpha=leaky_value, name="{}_act_s2".format(model_name))(conv_s2)

    conv_s3 = Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        kernel_initializer=initializer,
        kernel_regularizer=l2(l2_weight),
        padding="same",
        name="{}_conv_s3".format(model_name),
    )(conv_s)
    conv_s3 = BatchNormalization(name="{}_bn_s3".format(model_name))(conv_s3)
    conv_s3 = LeakyReLU(alpha=leaky_value, name="{}_act_s3".format(model_name))(conv_s3)

    # (HxW) x C
    conv_s1 = Reshape(
        (inputs.shape[1] * inputs.shape[2], inputs.shape[3]),
        input_shape=(inputs.shape[1], inputs.shape[2], inputs.shape[3]),
        name="{}_reshape_s1".format(model_name),
    )(conv_s1)

    # (HxW) x C
    conv_s2 = Reshape(
        (inputs.shape[1] * inputs.shape[2], inputs.shape[3]),
        input_shape=(inputs.shape[1], inputs.shape[2], inputs.shape[3]),
        name="{}_reshape_s2".format(model_name),
    )(conv_s2)

    # C x (HxW)
    conv_s2 = Permute((2, 1))(conv_s2)

    # (HxW) x (HxW)
    conv_s2 = Dot(axes=(1, 2), name="{}_dot_s2".format(model_name))([conv_s2, conv_s1])

    # perform on each row  (HxW) x (HxW)
    conv_s2 = Softmax(axis=-1, name="{}_softmax_s2".format(model_name))(conv_s2)

    # (HxW) x C
    conv_s3 = Reshape(
        (inputs.shape[1] * inputs.shape[2], inputs.shape[3]),
        input_shape=(inputs.shape[1], inputs.shape[2], inputs.shape[3]),
        name="{}_reshape_s3".format(model_name),
    )(conv_s3)

    # (HxW) x C
    conv_s3 = Dot(axes=(1, 2), name="{}_dot_s3".format(model_name))([conv_s3, conv_s2])

    # H x W x C
    conv_s3 = Reshape(
        (inputs.shape[1], inputs.shape[2], inputs.shape[3]),
        input_shape=(inputs.shape[1] * inputs.shape[2], inputs.shape[3]),
        name="{}_reshapeS_final".format(model_name),
    )(conv_s3)

    conv_s3 = Attention(name="{}_weights_multiply_s".format(model_name))(conv_s3)

    add_s = Add(name="{}_addS".format(model_name))([conv_s, conv_s3])

    # channel attention module
    conv_c = Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        kernel_initializer=initializer,
        kernel_regularizer=l2(l2_weight),
        padding="same",
        name="{}_conv_c".format(model_name),
    )(inputs)
    conv_c = BatchNormalization(name="{}_bn_c".format(model_name))(conv_c)
    conv_c = LeakyReLU(alpha=leaky_value, name="{}_act_c".format(model_name))(conv_c)

    conv_c1 = Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        kernel_initializer=initializer,
        kernel_regularizer=l2(l2_weight),
        padding="same",
        name="{}_conv_c1".format(model_name),
    )(conv_c)
    conv_c1 = BatchNormalization(name="{}_bn_c1".format(model_name))(conv_c1)
    conv_c1 = LeakyReLU(alpha=leaky_value, name="{}_act_c1".format(model_name))(conv_c1)

    conv_c2 = Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        kernel_initializer=initializer,
        kernel_regularizer=l2(l2_weight),
        padding="same",
        name="{}_conv_c2".format(model_name),
    )(conv_c)
    conv_c2 = BatchNormalization(name="{}_bn_c2".format(model_name))(conv_c2)
    conv_c2 = LeakyReLU(alpha=leaky_value, name="{}_act_c2".format(model_name))(conv_c2)

    conv_c3 = Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        kernel_initializer=initializer,
        kernel_regularizer=l2(l2_weight),
        padding="same",
        name="{}_conv_c3".format(model_name),
    )(conv_c)
    conv_c3 = BatchNormalization(name="{}_bn_c3".format(model_name))(conv_c3)
    conv_c3 = LeakyReLU(alpha=leaky_value, name="{}_act_c3".format(model_name))(conv_c3)

    # (HxW) x C
    conv_c1 = Reshape(
        (inputs.shape[1] * inputs.shape[2], inputs.shape[3]),
        input_shape=(inputs.shape[1], inputs.shape[2], inputs.shape[3]),
        name="{}_reshape_c1".format(model_name),
    )(conv_c1)

    # (HxW) x C
    conv_c2 = Reshape(
        (inputs.shape[1] * inputs.shape[2], inputs.shape[3]),
        input_shape=(inputs.shape[1], inputs.shape[2], inputs.shape[3]),
        name="{}_reshape_c2".format(model_name),
    )(conv_c2)

    # C x (HxW)
    conv_c2 = Permute((2, 1))(conv_c2)

    # C x C
    conv_c2 = Dot(axes=(1, 2), name="{}_dot_c2".format(model_name))([conv_c1, conv_c2])

    # perform on each row  C x C
    conv_c2 = Softmax(axis=-1, name="{}_softmax_c2".format(model_name))(conv_c2)

    # (HxW) x C
    conv_c3 = Reshape(
        (inputs.shape[1] * inputs.shape[2], inputs.shape[3]),
        input_shape=(inputs.shape[1], inputs.shape[2], inputs.shape[3]),
        name="{}_reshape_c3".format(model_name),
    )(conv_c3)

    # C x (HxW)
    conv_c3 = Permute((2, 1))(conv_c3)

    # C x (HxW)
    conv_c3 = Dot(axes=(1, 2), name="{}_dot_c3".format(model_name))([conv_c3, conv_c2])

    # (HxW) x C
    conv_c3 = Permute((2, 1))(conv_c3)

    # H x W x C
    conv_c3 = Reshape(
        (inputs.shape[1], inputs.shape[2], inputs.shape[3]),
        input_shape=(inputs.shape[1] * inputs.shape[2], inputs.shape[3]),
        name="{}_reshapeC_final".format(model_name),
    )(conv_c3)

    conv_c3 = Attention(name="{}_weights_multiply_c".format(model_name))(conv_c3)

    add_c = Add(name="{}_addC".format(model_name))([conv_c, conv_c3])

    return Add(name="{}_final_add".format(model_name))([add_s, add_c])
