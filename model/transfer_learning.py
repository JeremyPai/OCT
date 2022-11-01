from xml.etree.ElementInclude import include
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet_v2 import ResNet50V2


def TransferModel(input_tensor):
    # VGG16
    base_model = VGG16(
        include_top=True,
        weights="imagenet",
        input_tensor=input_tensor,
        classes=3,
        classifier_activation="softmax",
    )

    for layer in base_model.layers[15:]:
        layer.trainable = True

    # ResNet50V2
    base_model = ResNet50V2(
        include_top=True,
        weights="imagenet",
        input_tensor=input_tensor,
        classes=3,
        classifier_activation="softmax",
    )

    for layer in base_model.layers[-10:]:
        layer.trainable = True

    return base_model
