from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adagrad
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Train:
    def __init__(self, dataset, model, model_name):
        self.dataset = dataset

        (self.train_data, self.train_label, self.input_shape),
        (self.validate_data, self.validate_label),
        (self.test_data, self.test_label) = self.dataset

        self.model = model
        self.model_name = model_name
        self.epochs = 10000
        self.batch_size = 32

        self.datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,
            vertical_flip=False,
            fill_mode="nearest",
        )

        # optimizer
        self.optimizer = SGD(learning_rate=0.0001, momentum=0.9)
        # self.optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
        # self.optimizer = RMSprop(
        #     learning_rate=0.0001, rho=0.9, momentum=0.0, epsilon=1e-07
        # )
        # self.optimizer = Adagrad(
        #     learning_rate=0.0001, initial_accumulator_value=0.1, epsilon=1e-07
        # )

        # callback
        self.EarlyStopping_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=15,
            verbose=1,
            mode="auto",
            restore_best_weights=True,
        )

        self.ReduceLROnPlateau_callback = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1,
            mode="auto",
            cooldown=30,
            min_lr=0,
        )

        self.history_list = []

    def start_training(self):
        print("start training...")

        # train the model again, this time fine-tuning *both* the final set of
        # CONV layers along with our set of FC layers
        print("transfer learning only need fine-tuning")

        self.model.compile(
            optimizer=self.optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        history = self.model.fit(
            self.datagen.flow(
                self.train_data,
                self.train_label,
                batch_size=self.batch_size,
                shuffle=True,
            ),
            epochs=self.epochs,
            shuffle=True,
            validation_data=(self.validate_data, self.validate_label),
            steps_per_epoch=self.train_data.shape[0] // self.batch_size,
            callbacks=[
                self.EarlyStopping_callback,
                self.ReduceLROnPlateau_callback,
            ],
        )

        self.history_list.append(history)

        loss, acc = self.model.evaluate(
            self.test_data, self.test_label, batch_size=self.batch_size, verbose=2
        )

        self.model.save(self.model_name)

        print("training result...")
        print("loss: {:.4f}".format(loss))
        print("accuracy: {:.2f}%".format(100 * acc))
        print("save model complete!")

        self.plot_learning_curve()

    def plot_learning_curve(self):
        history_accuracy = []
        history_val_accuracy = []
        history_loss = []
        history_val_loss = []

        for hist in self.history_list:
            history_accuracy = history_accuracy + hist.history["accuracy"]
            history_val_accuracy = history_val_accuracy + hist.history["val_accuracy"]
            history_loss = history_loss + hist.history["loss"]
            history_val_loss = history_val_loss + hist.history["val_loss"]

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(
            np.arange(1, len(history_accuracy) + 1), history_accuracy, label="Train"
        )
        plt.plot(
            np.arange(1, len(history_val_accuracy) + 1),
            history_val_accuracy,
            label="Validation",
        )
        plt.title("Training and Validation Accuracy")
        plt.ylabel("Accuracy")
        plt.legend(loc="best")

        plt.subplot(2, 1, 2)
        plt.plot(np.arange(1, len(history_loss) + 1), history_loss, label="Training")
        plt.plot(
            np.arange(1, len(history_val_loss) + 1),
            history_val_loss,
            label="Validation",
        )
        plt.title("Training and Validation Loss")
        plt.ylabel("Cross Entropy loss")
        plt.xlabel("Epoch")
        plt.legend(loc="best")
        plt.show()

        plt.savefig(self.model_name + " Learning Curves.png")
