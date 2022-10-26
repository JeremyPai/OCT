import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming

"""""" """""" """""" """""" """""" """""" """
    author: Jeremy Pai

    影像重建 (Image Reconstruction)
    
    filePath: OCT 干涉訊號的路徑 (Single_offline data)
    
    backgroundPath: background 的路徑，如果沒有要作減去 background 的處理，就將 background_subtraction 
                     設為 False
                     
    numOfBackground: 有幾張 background，只有 background_subtraction = True 才需設定
    
    coeffOfSecondOrderDispersion: 做色散補償時的參數，需依情況做調整
    
    numOfALines: 一張 B-scan 中有幾條 A-line
    
    window: 使用 Window 擷取需要的影像區域
    
    pixelOfRawData: A-line 中有幾個 pixels
    
    pixelOfCropData: A-line 中捨棄的 pixels 數
    
    pixelAfterZeroPadding: 作 zero padding 後 A-line 的 pixels 數
    
    numOfImages: 一組 C-scan 中的 B-scan 數
    
    beginOfImages, endOfImages: 選取要重建的 OCT 影像
""" """""" """""" """""" """""" """""" """"""


class ReconstructImage:
    def __init__(self, config):
        self.filePath = config["filePath"]
        self.backgroundPath = config["backgroundPath"]
        self.backgroundSubtraction = config["backgroundSubtraction"]
        self.numOfBackground = config["numOfBackground"]
        self.coeffOfSecondOrderDispersion = config["coeffOfSecondOrderDispersion"]
        self.numOfALines = config["numOfALines"]
        self.windowRow = config["window"][0]
        self.windowCol = config["window"][1]
        self.pixelOfRawData = config["pixelOfRawData"]
        self.pixelOfCropData = config["pixelOfCropData"]
        self.pixelAfterZeroPadding = config["pixelAfterZeroPadding"]
        self.numOfImages = config["numOfImages"]
        self.beginOfImages = config["beginOfImages"]
        self.endOfImages = config["endOfImages"]

    def reconstruct(self):
        for dirPath, dirNames, fileNames in os.walk(self.filePath):
            if not dirNames:
                pathReconstruct = os.path.join(dirPath, "reconstructed")

                if not os.path.exists(pathReconstruct):
                    os.mkdir(pathReconstruct)

                os.chdir(pathReconstruct)

                for file in fileNames:
                    if file.endswith(".dat"):
                        # OCT = np.fromfile(os.path.join(dirPath, file), dtype='>i2')
                        OCT = np.fromfile(os.path.join(dirPath, file), dtype=">u2")
                        OCT = OCT.astype(np.float64)

                        # background subtraction
                        meanOfBackground = self.prepareBackground()
                        OCT = np.subtract(OCT, meanOfBackground)

                        Bscan = np.zeros(
                            (self.pixelAfterZeroPadding, self.numOfALines)
                        ).astype(np.float64)

                        for index in range(self.numOfALines):
                            # delete 200 pixels in the front and in the back
                            # interference = OCT[
                            #     self.pixelOfRawData * index
                            #     + 2
                            #     + 200 : self.pixelOfRawData * (index + 1)
                            #     + 2
                            #     - 200
                            # ].astype(np.float64)

                            interference = OCT[
                                self.pixelOfRawData * index
                                + 2 : self.pixelOfRawData * (index + 1)
                                + 2
                            ].astype(np.float64)

                            # dispersion compensation
                            dispersion = self.prepareDispersionCompensation()
                            interference = np.multiply(interference, dispersion)

                            # window function
                            # window = self.prepareWindowFunc()
                            # interference = np.multiply(interference, window)

                            # zero padding
                            # zeroToPad = 1 << len(interference).bit_length()
                            # zeroToPad = zeroToPad - len(interference)

                            # zeroToPad = (
                            #     self.pixelAfterZeroPadding - self.pixelOfCropData
                            # )

                            # interference = np.pad(
                            #     interference,
                            #     (0, zeroToPad),
                            #     mode="constant",
                            #     constant_values=0,
                            # )

                            # Fourier transform
                            interference = np.fft.fft(interference).astype(
                                np.complex128
                            )
                            interference = np.fft.fftshift(interference)

                            # only amplitude needed
                            interference = np.abs(interference)

                            Bscan[:, index] = interference

                        Bscan = Bscan[
                            self.windowRow[0] : self.windowRow[1],
                            self.windowCol[0] : self.windowCol[1],
                        ]

                        Bscan = 20 * np.log10(Bscan)
                        # Bscan = Bscan - np.min(Bscan)
                        Bscan = np.rot90(Bscan, 2)

                        # calculate information entropy
                        temp = np.divide(Bscan, np.sum(Bscan))
                        information_entropy = -np.sum(np.multiply(temp, np.log10(temp)))
                        print(
                            "After dispersion compensation, information entropy is: ",
                            information_entropy,
                        )

                        # extract number of C-scan image
                        num = ""
                        for c in file.split(".")[0]:
                            if c.isdigit():
                                num = num + c

                        # save figure and image
                        plt.imsave(
                            "{}_image.png".format(num),
                            Bscan,
                            cmap="gray",
                            vmin=65,
                            vmax=120,
                        )  # OCT: [75,120]

                        np.save("{}_image".format(num), Bscan, allow_pickle=False)

    def prepareBackground(self):
        # calculate mean value of background
        meanOfBackground = np.zeros(
            (self.numOfALines * self.pixelOfRawData + 2), dtype=np.float64
        )

        for num in range(1, self.numOfBackground + 1):
            pathBackground = os.path.join(
                self.backgroundPath, "Single_offline%d.dat" % num
            )

            # big-endian 16-bit signed integer
            # fileBackground = np.fromfile(pathBackground, dtype='>i2')

            # big-endian 16-bit unsigned integer
            fileBackground = np.fromfile(pathBackground, dtype=">u2")
            meanOfBackground = np.add(meanOfBackground, fileBackground)

        meanOfBackground = np.divide(meanOfBackground, self.numOfBackground)

        return meanOfBackground

    def prepareDispersionCompensation(self):
        # dispersion = np.ones((self.pixelOfCropData), dtype=np.complex128)
        dispersion = np.ones((self.pixelOfRawData), dtype=np.complex128)

        if self.coeffOfSecondOrderDispersion != 0:
            for index in range(len(dispersion)):
                temp = complex(
                    0,
                    self.coeffOfSecondOrderDispersion
                    * (index / (len(dispersion) - 1)) ** 2,
                )
                dispersion[index] = np.exp(temp)

        return dispersion

    def prepareWindowFunc(self):
        return hamming(self.pixelOfCropData, sym=True).astype(np.float64)
