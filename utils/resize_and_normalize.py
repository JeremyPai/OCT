import os
import numpy as np
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt


class ResizeAndNormalize:
    def __init__(self, config):
        self.filePath = config["filePath"]

    def process(self):
        for dirPath, dirNames, fileNames in os.walk(self.filePath):
            if "denoise" in dirNames:
                pathDeep = os.path.join(dirPath, "deep_01")

                if not os.path.exists(pathDeep):
                    os.mkdir(pathDeep)

            if os.path.basename(dirPath) == "denoise":
                os.chdir(pathDeep)
                for file in fileNames:
                    if file.endswith(".npy"):
                        image = np.load(os.path.join(dirPath, file))

                        for z in range(image.shape[0]):
                            for x in range(image.shape[1]):
                                if image[z, x] >= 120:
                                    image[z, x] = 120

                                elif image[z, x] < 65:
                                    image[z, x] = 65

                        # resize
                        resizeImage = zoom(image, [0.4, 0.072159], prefilter=False)

                        # normalize image from 0 to 1
                        resizeImage = (resizeImage - np.min(resizeImage)) / (
                            np.max(resizeImage) - np.min(resizeImage)
                        )

                        plt.imsave(
                            file.split("_")[0] + "_image.png", resizeImage, cmap="gray"
                        )
                        np.save(
                            file.split("_")[0] + "_image",
                            resizeImage,
                            allow_pickle=False,
                        )
