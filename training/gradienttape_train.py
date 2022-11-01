from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adagrad
from tensorflow.keras.utils import Progbar
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


class Train:
    def __init__(self, dataset, model, model_name):
        self.dataset = dataset

        (self.train_data, self.train_label, self.input_shape),
        (self.validate_data, self.validate_label),
        (self.test_data, self.test_label) = self.dataset

        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.train_data, self.train_label)
        )
        self.train_dataset = self.train_dataset.shuffle(
            buffer_size=25000, reshuffle_each_iteration=True
        )

        self.batch_size = 32
        self.validate_dataset = tf.data.Dataset.from_tensor_slices(
            (self.validate_data, self.validate_label)
        ).batch(self.batch_size)

        self.model = model
        self.model_name = model_name
        self.epochs = 10000

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

        # TensorBoard
        train_log_dir = os.path.join("logs", self.model_name, "train")
        validate_log_dir = os.path.join("logs", self.model_name, "validation")

        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.validate_summary_writer = tf.summary.create_file_writer(validate_log_dir)

        # optimizer
        self.optimizer = SGD(learning_rate=0.0001, momentum=0.9)
        # self.optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
        # self.optimizer = RMSprop(
        #     learning_rate=0.0001, rho=0.9, momentum=0.0, epsilon=1e-07
        # )
        # self.optimizer = Adagrad(
        #     learning_rate=0.0001, initial_accumulator_value=0.1, epsilon=1e-07
        # )

        # loss function
        self.loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        # select metrics to measure the loss and the accuracy of the model
        self.train_loss = tf.keras.metrics.CategoricalCrossentropy(name="train_loss")
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(
            name="train_accuracy"
        )

        self.val_loss = tf.keras.metrics.CategoricalCrossentropy(name="val_loss")
        self.val_accuracy = tf.keras.metrics.CategoricalAccuracy(name="val_accuracy")

    @tf.function
    def train_step(self, images, labels, optimizer):
        with tf.GradientTape() as tape:
            # prediction
            predictions = self.model(images, training=True)

            # loss
            loss = self.loss_function(labels, predictions)

        # calculate the gradients using our tape and then update the model weights
        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss.update_state(labels, predictions)
        self.train_accuracy.update_state(labels, predictions)

    @tf.function
    def validate_step(self, images, labels):
        v_predictions = self.model(images, training=False)
        # v_loss = self.loss_function(labels, v_predictions)

        self.val_loss.update_state(labels, v_predictions)
        self.val_accuracy.update_state(labels, v_predictions)

    def start_training(self):
        print("start training...")

        best_val_loss = 1000
        patience_earlystop = 15
        accumulate_earlystop = 0
        steps_per_epoch = self.train_data.shape[0] // self.batch_size

        # patience_lr = 5
        # accumulate_lr = 0
        # cooldown = 30
        # factor = 0.5

        # learning_rate = 0.0001

        attention_weights = []

        for epoch in range(self.epochs):
            # reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.val_loss.reset_states()
            self.val_accuracy.reset_states()

            # initialize the progress bar
            progbar = Progbar(target=steps_per_epoch)

            tf.print("Epoch {}/{}".format(epoch + 1, self.epochs))

            for index, (images, labels) in enumerate(
                self.train_dataset.batch(self.batch_size)
            ):
                if index == steps_per_epoch:
                    break

                images = self.datagen.flow(
                    images, batch_size=self.batch_size, shuffle=False
                )[0]

                self.train_step(images, labels, self.optimizer)

                # This will update the progress bar graph
                progbar.update(index + 1)

            with self.train_summary_writer.as_default():
                tf.summary.scalar("loss", self.train_loss.result(), step=epoch)
                tf.summary.scalar("accuracy", self.train_accuracy.result(), step=epoch)

            for val_images, val_labels in self.validate_dataset:
                self.validate_step(val_images, val_labels)

            with self.validate_summary_writer.as_default():
                tf.summary.scalar("loss", self.val_loss.result(), step=epoch)
                tf.summary.scalar("accuracy", self.val_accuracy.result(), step=epoch)

            template = " - loss: {:.4f} - accuracy: {:.4f} - val loss: {:.4f} - val accuracy: {:.4f}"
            tf.print(
                template.format(
                    self.train_loss.result(),
                    self.train_accuracy.result(),
                    self.val_loss.result(),
                    self.val_accuracy.result(),
                )
            )

            # attention weights
            attention_weights.append(
                (
                    self.model.get_layer(name="attention_18").get_weights(),
                    self.model.get_layer(name="attention_19").get_weights(),
                    self.model.get_layer(name="attention_20").get_weights(),
                    self.model.get_layer(name="attention_21").get_weights(),
                )
            )

            if best_val_loss < self.val_loss.result():
                accumulate_earlystop += 1
                accumulate_lr += 1

                if accumulate_earlystop == patience_earlystop:
                    tf.print("Early Stopping!!")
                    break

                # elif epoch > cooldown and accumulate_lr >= patience_lr:
                #     learning_rate *= factor
                #     tf.print("Change Learning rate to {}".format(learning_rate))

            else:
                best_val_loss = self.val_loss.result()
                accumulate_earlystop = 0
                accumulate_lr = 0

                self.model.save(self.model_name)
                tf.print("model save to {}".format(self.model_name))

            self.train_dataset = self.train_dataset.shuffle(buffer_size=25000)

        self.model = load_model(filepath=self.model_name)

        loss, acc = self.model.evaluate(
            self.test_data, self.test_label, batch_size=self.batch_size, verbose=2
        )

        print("training result...")
        print("loss: {:.4f}".format(loss))
        print("accuracy: {:.2f}%".format(100 * acc))

    def get_batch_size(self):
        return self.batch_size
