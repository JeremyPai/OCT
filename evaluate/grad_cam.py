# Class Activation Map
# https://www.sicara.ai/blog/2019-08-28-interpretability-deep-learning-tensorflow

from tensorflow.keras import Model
import tensorflow as tf
import cv2
import numpy as np


class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class activation map, and the layer to
        # be used when visualizing the class avtivation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName

        # if the layer name is None, attempt to automatically find the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network by looping over the layers of
        # the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.Name

        # otherwise, we could not find a 4D layers so the GradCAM algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-30):
        # construct our gradient model by supplying
        # (1) the inputs to our pre-trained model
        # (2) the output of the (presumably) final 4D layer in the network
        # (3) the output of the softmax activations from the model
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[
                self.model.get_layer(self.layerName[0]).output,
                self.model.get_layer(self.layerName[1]).output,
            ],
        )

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # convert input image shape to 4D
            image = image[np.newaxis, :]

            # cast the image tensor to a float32 data type, pass the image through the gradient model, and
            # grab the loss associated with the specific class index
            inputs = tf.cast(image, tf.float32)

            convOutputs, predictions = gradModel(inputs)
            loss = predictions[:, self.classIdx]

        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        # Keep in mind that both castConvOutputs and castGrads contain only values of 1's and 0's
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        # the convolution and guided gradients have a batch dimension (which we don't need) so let's
        # grab the volume itself and discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them as weights, compute the ponderation of
        # the filters with respect to the weights
        weights = tf.math.reduce_mean(guidedGrads, axis=(0, 1))
        CAM = tf.math.reduce_sum(tf.math.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize the output class activation map to
        # match the input image dimensions
        heatmap = cv2.resize(
            CAM.numpy(), (image.shape[2], image.shape[1]), interpolation=cv2.INTER_CUBIC
        )

        # normalize the heatmap such that all values lie in the range [0,1], scale the resulting values
        # to the range [0,255], and then convert to an unsigned 8-bit integer
        heatmap = (heatmap - np.min(heatmap)) / (
            (np.max(heatmap) - np.min(heatmap)) + eps
        )

        # return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(
        self,
        heatmap,
        image,
        ifGray=False,
        channelRelate=False,
        alpha=0.5,
        colormap=cv2.COLORMAP_VIRIDIS,
    ):
        """
        channelRelate: specifying each channel containing different features
        ex. first channel represents intensity image
            second channel represents attenuation image

        """
        # apply the supplied color map to the heatmap
        heatmap_color = cv2.applyColorMap(heatmap, colormap)

        # heatmap_color[np.where(heatmap<0.2)] = 0

        # convert data type of image, which is numpy array, to cv2 image data type
        if image.shape[-1] == 1 and ifGray == True and channelRelate == False:
            # must be a grayscale image
            image = np.uint8(255 * image)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # overlay the heatmap on the input image
            output = cv2.addWeighted(image, alpha, heatmap_color, 1 - alpha, 0)

        elif image.shape[-1] > 1 and ifGray == True and channelRelate == True:
            # must be a grayscale image combined with other channels representing other features
            # output should be a list containing heatmap of each channel
            output = []

            for channel in image.shape[-1]:
                image_channel = np.uint8(255 * image[:, :, channel][:, :, np.newaxis])
                image_channel = cv2.cvtColor(image_channel, cv2.COLOR_GRAY2BGR)

                # overlay the heatmap on the input image
                output.append(
                    cv2.addWeighted(image_channel, alpha, heatmap_color, 1 - alpha, 0)
                )

        elif image.shape[-1] > 1 and ifGray == False and channelRelate == False:
            # must be a RGB image
            # overlay the heatmap on the input image
            output = cv2.addWeighted(
                cv2.cvtColor((255 * image).astype("uint8"), cv2.COLOR_RGB2BGR),
                alpha,
                heatmap_color,
                1 - alpha,
                0,
            )

        # return a 2-tuple of the color mapped heatmap and the output, overlaid image
        return (heatmap_color, output)
