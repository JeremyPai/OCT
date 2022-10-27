from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import random


class BrainTumorDataset:
    def __init__(self):
        self.filePath = "D:\\brain_tumor"
        self.numChannel = 1

        # augmenting training data to augment_ratio times bigger (1 means the same)
        self.augmentRatio = 2

        self.Glioma_intensity_train = []
        self.Glioma_intensity_validate = []
        self.Glioma_intensity_test = []

        self.Lymphoma_intensity_train = []
        self.Lymphoma_intensity_validate = []
        self.Lymphoma_intensity_test = []

        self.Normal_intensity_train = []
        self.Normal_intensity_validate = []
        self.Normal_intensity_test = []

        self.load_images()

    def load_images(self):
        os.chdir(self.filePath)
        for dirPath, dirNames, fileNames in os.walk(self.filePath):
            if not dirNames:
                if os.path.basename(dirPath) == "deep_01":
                    for file in fileNames:
                        if file.endswith(".npy"):
                            image = np.load(os.path.join(dirPath, file))

                            if "Glioma" in dirPath:
                                if "train" in dirPath:
                                    self.Glioma_intensity_train.append(image)

                                elif "validate" in dirPath:
                                    self.Glioma_intensity_validate.append(image)

                                elif "test" in dirPath:
                                    self.Glioma_intensity_test.append(image)

                            elif "Lymphoma" in dirPath:
                                if "train" in dirPath:
                                    self.Lymphoma_intensity_train.append(image)

                                elif "validate" in dirPath:
                                    self.Lymphoma_intensity_validate.append(image)

                                elif "test" in dirPath:
                                    self.Lymphoma_intensity_test.append(image)

                            elif "Normal" in dirPath:
                                if "train" in dirPath:
                                    self.Normal_intensity_train.append(image)

                                elif "validate" in dirPath:
                                    self.Normal_intensity_validate.append(image)

                                elif "test" in dirPath:
                                    self.Normal_intensity_test.append(image)

    def prepare_images(self):
        (train_data, train_label, input_shape) = self.prepare_train()
        (validate_data, validate_label) = self.prepare_validate()
        (test_data, test_label) = self.prepare_test()

        return (
            (train_data, train_label, input_shape),
            (validate_data, validate_label),
            (test_data, test_label),
        )

    def prepare_train(self):
        """""" """""" """""" """""" """""" """""
                training data

        """ """""" """""" """""" """""" """""" ""
        train_data = np.zeros(
            (
                (3 + 3 * (self.augmentRatio - 1)) * len(self.Lymphoma_intensity_train),
                self.Glioma_intensity_train[0].shape[0],
                self.Glioma_intensity_train[0].shape[1],
                self.numChannel,
            )
        )

        Glioma_random_pick_index = random.sample(
            list(range(len(self.Glioma_intensity_train))),
            k=len(self.Lymphoma_intensity_train),
        )
        Normal_random_pick_index = random.sample(
            list(range(len(self.Normal_intensity_train))),
            k=len(self.Lymphoma_intensity_train),
        )

        for index in range(3 * len(self.Lymphoma_intensity_train)):
            if index < len(self.Lymphoma_intensity_train):
                train_data[index, :, :, 0] = self.Normal_intensity_train[
                    Normal_random_pick_index[index]
                ]
                train_data[
                    index + 3 * len(self.Lymphoma_intensity_train), :, :, 0
                ] = np.flip(train_data[index, :, :, 0], axis=1)

            elif index >= len(self.Lymphoma_intensity_train) and index < 2 * len(
                self.Lymphoma_intensity_train
            ):
                train_data[index, :, :, 0] = self.Glioma_intensity_train[
                    Glioma_random_pick_index[index - len(self.Lymphoma_intensity_train)]
                ]
                train_data[
                    index + 3 * len(self.Lymphoma_intensity_train), :, :, 0
                ] = np.flip(train_data[index, :, :, 0], axis=1)

            else:
                train_data[index, :, :, 0] = self.Lymphoma_intensity_train[
                    index - 2 * len(self.Lymphoma_intensity_train)
                ]
                train_data[
                    index + 3 * len(self.Lymphoma_intensity_train), :, :, 0
                ] = np.flip(train_data[index, :, :, 0], axis=1)

        train_label = np.concatenate(
            (
                np.zeros(len(self.Lymphoma_intensity_train)),
                np.ones(len(self.Lymphoma_intensity_train)),
                2 * np.ones(len(self.Lymphoma_intensity_train)),
            ),
            axis=0,
        )

        train_label = np.concatenate((train_label, train_label), axis=0)

        train_label = to_categorical(train_label)

        input_shape = (
            self.Glioma_intensity_train[0].shape[0],
            self.Glioma_intensity_train[0].shape[1],
            self.numChannel,
        )

        return (train_data, train_label, input_shape)

    def prepare_validate(self):
        """""" """""" """""" """""" """""" """""
                validation data

        """ """""" """""" """""" """""" """""" ""
        validate_data = np.zeros(
            (
                3 * len(self.Lymphoma_intensity_validate),
                self.Glioma_intensity_validate[0].shape[0],
                self.Glioma_intensity_validate[0].shape[1],
                self.numChannel,
            )
        )

        Normal_random_pick_index = random.sample(
            list(range(len(self.Normal_intensity_validate))),
            k=len(self.Lymphoma_intensity_validate),
        )
        Glioma_random_pick_index = random.sample(
            list(range(len(self.Glioma_intensity_validate))),
            k=len(self.Lymphoma_intensity_validate),
        )

        for index in range(3 * len(self.Lymphoma_intensity_validate)):
            if index < len(self.Lymphoma_intensity_validate):
                validate_data[index, :, :, 0] = self.Normal_intensity_validate[
                    Normal_random_pick_index[index]
                ]

            elif index >= len(self.Lymphoma_intensity_validate) and index < 2 * len(
                self.Lymphoma_intensity_validate
            ):
                validate_data[index, :, :, 0] = self.Glioma_intensity_validate[
                    Glioma_random_pick_index[
                        index - len(self.Lymphoma_intensity_validate)
                    ]
                ]

            else:
                validate_data[index, :, :, 0] = self.Lymphoma_intensity_validate[
                    index - 2 * len(self.Lymphoma_intensity_validate)
                ]

        validate_label = np.concatenate(
            (
                np.zeros(len(self.Lymphoma_intensity_validate)),
                np.ones(len(self.Lymphoma_intensity_validate)),
                2 * np.ones(len(self.Lymphoma_intensity_validate)),
            ),
            axis=0,
        )

        validate_label = to_categorical(validate_label)

        return (validate_data, validate_label)

    def prepare_test(self):
        """""" """""" """""" """""" """""" """""
                testing data

        """ """""" """""" """""" """""" """""" ""
        test_data = np.zeros(
            (
                len(self.Normal_intensity_test)
                + len(self.Glioma_intensity_test)
                + len(self.Lymphoma_intensity_test),
                self.Glioma_intensity_test[0].shape[0],
                self.Glioma_intensity_test[0].shape[1],
                self.numChannel,
            )
        )

        for index in range(
            len(self.Normal_intensity_test)
            + len(self.Glioma_intensity_test)
            + len(self.Lymphoma_intensity_test)
        ):
            if index < len(self.Normal_intensity_test):
                test_data[index, :, :, 0] = self.Normal_intensity_test[index]

            elif index >= len(self.Normal_intensity_test) and index < len(
                self.Normal_intensity_test
            ) + len(self.Glioma_intensity_test):
                test_data[index, :, :, 0] = self.Glioma_intensity_test[
                    index - len(self.Normal_intensity_test)
                ]

            else:
                test_data[index, :, :, 0] = self.Lymphoma_intensity_test[
                    index
                    - len(self.Normal_intensity_test)
                    - len(self.Glioma_intensity_test)
                ]

        test_label = np.concatenate(
            (
                np.zeros(len(self.Normal_intensity_test)),
                np.ones(len(self.Glioma_intensity_test)),
                2 * np.ones(len(self.Lymphoma_intensity_test)),
            ),
            axis=0,
        )

        test_label = to_categorical(test_label)

        return (test_data, test_label)
