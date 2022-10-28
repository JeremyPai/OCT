from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    LeakyReLU,
)
from tensorflow.keras.regularizers import l2


def residual_unit(
    inputs,
    filters=64,
    leaky_value=0,
    initializer="glorot_uniform",
    regularizer=l2(0.01),
    downsample=False,
    first_block=False,
    name="residual_unit",
):
    if first_block == False:
        x = BatchNormalization(name="{}_bn1".format(name))(inputs)
        x = LeakyReLU(alpha=leaky_value, name="{}_act1".format(name))(x)

    else:
        x = inputs

    if downsample == False:
        residual = x
        x = Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            padding="same",
            name="{}_conv1".format(name),
        )(x)

    else:
        residual = Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=(2, 2),
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            padding="same",
            name="{}_conv_resid".format(name),
        )(x)
        x = Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            padding="same",
            name="{}_conv1".format(name),
        )(x)

    x = BatchNormalization(name="{}_bn2".format(name))(x)
    x = LeakyReLU(alpha=leaky_value, name="{}_act2".format(name))(x)
    x = Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        padding="same",
        name="{}_conv2".format(name),
    )(x)

    return (x, residual)
