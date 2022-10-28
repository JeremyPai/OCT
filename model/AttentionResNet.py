from attention_unit import attention_block
from residual_unit import residual_unit
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    LeakyReLU,
    Add,
    Multiply,
    GlobalAveragePooling2D,
    Dense,
    Activation,
    MaxPooling2D,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Input, Model
import tensorflow as tf


def AttentionResNet(
    inputs,
    layers=(3, 4, 6, 3),
    filters=(64, 128, 256, 512),
    leaky_value=0,
    num_class=1000,
    initializer="glorot_uniform",
    l2_weight=0.05,
):
    x = Conv2D(
        filters=filters[0],
        kernel_size=(7, 7),
        strides=(2, 2),
        kernel_initializer=initializer,
        padding="same",
    )(inputs)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

    for num in range(layers[0]):
        if num == 0:
            x, residual = residual_unit(
                x,
                filters=filters[0],
                leaky_value=leaky_value,
                initializer=initializer,
                regularizer=l2(l2_weight),
                downsample=False,
                first_block=True,
                name="resid1_{}".format(num + 1),
            )

            x = Add(name="resid1_{}_add".format(num + 1))([x, residual])

        else:
            x, residual = residual_unit(
                x,
                filters=filters[0],
                leaky_value=leaky_value,
                initializer=initializer,
                regularizer=l2(l2_weight),
                downsample=False,
                first_block=False,
                name="resid1_{}".format(num + 1),
            )

            x = Add(name="resid1_{}_add".format(num + 1))([x, residual])

    for num in range(layers[1]):
        if num == 0:
            x, residual = residual_unit(
                x,
                filters=filters[1],
                leaky_value=leaky_value,
                initializer=initializer,
                regularizer=l2(l2_weight),
                downsample=True,
                first_block=False,
                name="resid2_{}".format(num + 1),
            )

            x = Add(name="resid2_{}_add".format(num + 1))([x, residual])

        else:
            x, residual = residual_unit(
                x,
                filters=filters[1],
                leaky_value=leaky_value,
                initializer=initializer,
                regularizer=l2(l2_weight),
                downsample=False,
                first_block=False,
                name="resid2_{}".format(num + 1),
            )

            x = Add(name="resid2_{}_add".format(num + 1))([x, residual])

    for num in range(layers[2]):
        if num == 0:
            x, residual = residual_unit(
                x,
                filters=filters[2],
                leaky_value=leaky_value,
                initializer=initializer,
                regularizer=l2(l2_weight),
                downsample=True,
                first_block=False,
                name="resid3_{}".format(num + 1),
            )

            x = Add(name="resid3_{}_add".format(num + 1))([x, residual])

        elif num == layers[2] - 1:
            x, residual = residual_unit(
                x,
                filters=filters[2],
                leaky_value=leaky_value,
                initializer=initializer,
                regularizer=l2(l2_weight),
                downsample=False,
                first_block=False,
                name="resid3_{}".format(num + 1),
            )

            # applying attention mechanism
            x = Add(name="resid3_{}_add".format(num + 1))([x, residual])

            feature_map = attention_block(
                inputs=x,
                filters=filters[2],
                leaky_value=leaky_value,
                initializer=initializer,
                l2_weight=l2_weight,
                model_name="attention_block_3",
            )

            x = Add(name="resid3_attention_{}".format(num + 1))(
                [Multiply()([feature_map, residual]), x, residual]
            )

        else:
            x, residual = residual_unit(
                x,
                filters=filters[2],
                leaky_value=leaky_value,
                initializer=initializer,
                regularizer=l2(l2_weight),
                downsample=False,
                first_block=False,
                name="resid3_{}".format(num + 1),
            )

            x = Add(name="resid3_{}_add".format(num + 1))([x, residual])

    for num in range(layers[3]):
        if num == 0:
            x, residual = residual_unit(
                x,
                filters=filters[3],
                leaky_value=leaky_value,
                initializer=initializer,
                regularizer=l2(l2_weight),
                downsample=True,
                first_block=False,
                name="resid4_{}".format(num + 1),
            )
            x = Add(name="resid4_{}_add".format(num + 1))([x, residual])

        elif num == layers[3] - 1:
            x, residual = residual_unit(
                x,
                filters=filters[3],
                leaky_value=leaky_value,
                initializer=initializer,
                regularizer=l2(l2_weight),
                downsample=False,
                first_block=False,
                name="resid4_{}".format(num + 1),
            )

            # applying attention mechanism
            x = Add(name="resid4_{}_add".format(num + 1))([x, residual])

            feature_map = attention_block(
                inputs=x,
                filters=filters[3],
                leaky_value=leaky_value,
                initializer=initializer,
                l2_weight=l2_weight,
                model_name="attention_block_4",
            )

            x = Add(name="resid4_attention_{}".format(num + 1))(
                [Multiply()([feature_map, residual]), x, residual]
            )

        else:
            x, residual = residual_unit(
                x,
                filters=filters[3],
                leaky_value=leaky_value,
                initializer=initializer,
                regularizer=l2(l2_weight),
                downsample=False,
                first_block=False,
                name="resid4_{}".format(num + 1),
            )
            x = Add(name="resid4_{}_add".format(num + 1))([x, residual])

    x = GlobalAveragePooling2D(name="avg_pool")(x)
    x = Dense(units=num_class, kernel_initializer=initializer, name="class_output")(x)
    outputs = Activation("softmax", name="class_output_act")(x)

    return Model(inputs=inputs, outputs=outputs)
