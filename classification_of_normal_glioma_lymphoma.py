from re import T
from dataset_preparation.dataset_preparation import BrainTumorDataset
from model.AttentionResNet import AttentionResNet
from model.transfer_learning import TransferModel
from training.keras_train import Train
from evaluate.confusion_matrix import ConfusionMatrix
from evaluate.t_sne import plot_tSNE
from evaluate.grad_cam import GradCAM
from tensorflow.keras import Input
import tensorflow as tf
import os
import numpy as np
import cv2
import time


if __name__ == "__main__":
    print("Tensorflow version: ", tf.__version__)
    print("Eager execution: ", tf.executing_eagerly())
    tf.keras.backend.set_floatx("float32")

    os.chdir("./")

    # prepare data
    dataset = BrainTumorDataset()
    data = dataset.prepare_images()
    train_data = data[0][0]
    train_label = data[0][1]
    input_shape = data[0][2]
    validate_data = data[1][0]
    validate_label = data[1][1]
    test_data = data[2][0]
    test_label = data[2][1]

    # prepare model to train
    # initializer: glorot_uniform, he_normal, he_uniform
    model = AttentionResNet(
        inputs=Input(shape=input_shape),
        layers=(1, 1, 2, 2),
        filters=(8, 16, 32, 64),
        leaky_value=0,
        num_class=3,
        initializer="he_normal",
        l2_weight=0.01,
    )

    # transfer learning
    # model = TransferModel(Input(shape=input_shape))

    model.summary()

    # train
    time_for_training = time.strftime("%m %d %Y %H.%M.%S")
    model_name = "AttentionResNet_{}.h5".format(time_for_training)

    tr = Train(data, model, model_name)
    tr.start_training()
    batch_size = tr.get_batch_size()

    # find the incorrect predicted label
    test_prediction = model.predict(test_data)
    incorrect_prediction = []
    for index in range(test_prediction.shape[0]):
        if np.argmax(test_prediction[index, :]) != np.argmax(test_label[index, :]):
            incorrect_prediction.append(index)

    print("incorrect prediction: ", incorrect_prediction)

    # evaluation
    # plot confusion matrix
    classes = ["Normal", "Glioma", "Lymphoma"]
    confusion = ConfusionMatrix(data, classes, model, model_name)
    confusion.plot_confusion_matrix()

    # plot t-SNE
    plot_tSNE(
        model,
        layername="avg_pool",
        input_data=test_data,
        input_label=test_label,
        input_predict=test_prediction,
        label_name=classes,
        batch_size=batch_size,
        modelname=model_name,
    )

    # use Grad-CAM to visualize possible features in the images
    i = 7000
    matrix_visualize = test_data[i]
    classIdx = np.argmax(test_prediction[i])

    # initialize our gradient class activation map and build the heatmap
    cam = GradCAM(
        model, classIdx=classIdx, layerName=["resid4_attention_2", "class_output"]
    )

    heatmap = cam.compute_heatmap(matrix_visualize)

    # overlay heatmap on top of the image
    (heatmap, cam_image) = cam.overlay_heatmap(
        (heatmap * 255).astype("uint8"),
        matrix_visualize[:, :, 0][:, :, np.newaxis],
        ifGray=True,
        channelRelate=False,
        alpha=0.5,
        colormap=cv2.COLORMAP_JET,
    )

    # make a image to put the text of label and accuracy
    black = np.zeros((matrix_visualize.shape[0], matrix_visualize.shape[1], 3)).astype(
        "uint8"
    )
    word_template = classes[classIdx] + ": {:.2f}%".format(
        test_prediction[i, classIdx] * 100
    )
    label_template = "Label: " + classes[np.argmax(test_label[i])]

    fontScale = 0.7
    fontThickness = 2
    textsize = cv2.getTextSize(
        word_template, cv2.FONT_HERSHEY_SIMPLEX, fontScale, fontThickness
    )[0]
    textX = int((black.shape[1] - textsize[0]) / 2)
    textY = int((black.shape[0] + textsize[1]) / 2)

    # cv2.putText(black, word_template, (textX,textY), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255,255,255), fontThickness)
    cv2.putText(
        black,
        word_template,
        (textX, textY + int(black.shape[1] / 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale,
        (255, 255, 255),
        fontThickness,
    )
    cv2.putText(
        black,
        label_template,
        (textX, textY - int(black.shape[1] / 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale,
        (255, 255, 255),
        fontThickness,
    )

    # display the original image and resulting heatmap and output image to our screen
    resized_width = matrix_visualize.shape[1] * 2
    resized_height = matrix_visualize.shape[0] * 2

    output1 = np.hstack(
        [
            cv2.resize(
                cv2.cvtColor(
                    np.uint8(matrix_visualize[:, :, 0] * 255), cv2.COLOR_GRAY2BGR
                ),
                (resized_width, resized_height),
            ),
            cv2.resize(heatmap, (resized_width, resized_height)),
        ]
    )
    output2 = np.hstack(
        [
            cv2.resize(cam_image, (resized_width, resized_height)),
            cv2.resize(black, (resized_width, resized_height)),
        ]
    )

    output = np.vstack([output1, output2])

    cv2.imshow("Output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(model_name + " CAM result {}.png".format(i), output)
