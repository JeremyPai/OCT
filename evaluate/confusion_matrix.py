from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ConfusionMatrix:
    def __init__(self, dataset, classes, model, model_name):
        self.dataset = dataset
        self.classes = classes

        (self.train_data, self.train_label, self.input_shape),
        (self.validate_data, self.validate_label),
        (self.test_data, self.test_label) = self.dataset

        self.model = model
        self.model_name = model_name

    def plot_confusion_matrix(self):
        test_prediction = self.model.predict(self.test_data)

        cnf_matrix = confusion_matrix(
            np.argmax(self.test_label, axis=1).reshape(-1, 1),
            np.argmax(test_prediction, axis=1).reshape(-1, 1),
            labels=range(len(self.classes)),
        )

        cnf_matrix_norm = np.around(
            cnf_matrix.astype("float") / np.sum(cnf_matrix, axis=1)[:, np.newaxis],
            decimals=2,
        )

        cnf_matrix_df = pd.DataFrame(
            cnf_matrix, index=self.classes, columns=self.classes
        )

        cnf_matrix_norm_df = pd.DataFrame(
            cnf_matrix_norm, index=self.classes, columns=self.classes
        )

        plt.figure()
        sns.heatmap(cnf_matrix_df, annot=True, fmt="d", cmap=plt.cm.Blues)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.title("Confusion Matrix")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(
            self.model_name + " Confusion matrix.png", dpi=350, bbox_inches="tight"
        )

        plt.figure()
        sns.heatmap(cnf_matrix_norm_df, annot=True, cmap=plt.cm.Blues)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.title("Normalized Confusion Matrix")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(
            self.model_name + " Normalized Confusion matrix.png",
            dpi=350,
            bbox_inches="tight",
        )
