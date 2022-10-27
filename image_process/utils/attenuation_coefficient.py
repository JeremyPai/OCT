import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class AttenuationCoeff:
    def __init__(self, config):
        self.filePath = config["filePath"]
        self.intralipidPath = config["intralipidPath"]
        self.sensitivityPath = config["sensitivityPath"]
        self.confocalPath = config["confocalPath"]
        self.dirNameToSolve = config["dirNameToSolve"]
        self.window = config["window"]

        # water: 1.34, tissue: 1.38, intralipid: 1.353
        self.refractionIndex = config["refractionIndex"]
        self.pixelSize = config["pixelSize"] * 10 ** (-3)

        # 2 mm
        self.focalPlane = config["focalPlane"]

    def attenuationCoeffImage(self):
        sensitivity = self.getSensitivity()
        confocal = self.getConfocal()
        noise, noiseMean = self.getNoise()

        for dirPath, dirNames, fileNames in os.walk(self.filePath):
            if self.dirNameToSolve in dirNames:
                pathAttenuation = os.path.join(dirPath, "attenuation map")

                if not os.path.exists(pathAttenuation):
                    os.mkdir(pathAttenuation)

            if os.path.basename(dirPath) == self.dirNameToSolve:
                os.chdir(pathAttenuation)
                for file in fileNames:
                    if file.endswith(".npy"):
                        imageDB = np.load(os.path.join(dirPath, file))

                        # unit dB turns into unit power
                        image_power = 10 ** (imageDB / 20)

                        # attenuation map
                        attenuation = np.zeros((imageDB.shape[0], imageDB.shape[1]))

                        for x in range(imageDB.shape[1]):
                            restorationFilter = np.zeros((imageDB.shape[0]))
                            estimated = np.zeros((imageDB.shape[0]))
                            accumulated = np.zeros((imageDB.shape[0]))

                            firstSignalAppear = 0

                            for z in reversed(range(imageDB.shape[0])):
                                if imageDB[z, x] == 0:
                                    break

                                else:
                                    restorationFilter[z] = (
                                        1 / (sensitivity[z] * confocal[z])
                                    ) * (
                                        image_power[z, x] ** 2
                                        / (image_power[z, x] ** 2 + noiseMean[z] ** 2)
                                    )

                                    estimated[z] = (
                                        image_power[z, x] * restorationFilter[z]
                                    )

                                    if firstSignalAppear == 1:
                                        accumulated[z] = (
                                            accumulated[z + 1] + estimated[z]
                                        )

                                        # unit: mm-1
                                        attenuation[z, x] = (
                                            estimated[z]
                                            * 10 ** (-3)
                                            / (2 * self.pixelSize * accumulated[z])
                                        )

                                    elif firstSignalAppear == 0:
                                        # if image_power[z,x] != 0 and image_power[z,x] > noise[z]:
                                        if image_power[z, x] != 0:
                                            accumulated[z] = estimated[z]

                                            # unit: mm-1
                                            attenuation[z, x] = (
                                                estimated[z]
                                                * 10 ** (-3)
                                                / (2 * self.pixelSize * accumulated[z])
                                            )

                                            firstSignalAppear = 1

                        # figure
                        # fig, ax = plt.subplots()
                        # ax1 = ax.imshow(attenuation, cmap='gray', vmin=0, vmax=4)
                        # ax.set_title('attenuation image ' + file.split('_')[0])
                        # ax.set_xlabel('pixel')
                        # ax.set_ylabel('pixel')
                        # ax.set_aspect(aspect=9.44)
                        # fig.colorbar(ax1)

                        plt.imsave(
                            file.split("_")[0] + "_attenuation.png",
                            attenuation,
                            cmap="gray",
                            vmin=0,
                            vmax=4,
                        )

                        np.save(
                            file.split("_")[0] + "_attenuation",
                            attenuation,
                            allow_pickle=False,
                        )

    def getSensitivity(self):
        # read sensitivity fall-off function
        fileSensitivity = os.path.join(
            self.sensitivityPath, "Sensitivity fall-off function.xlsx"
        )
        sensitivity = pd.read_excel(fileSensitivity)
        sensitivity = sensitivity["interpolation value"].to_numpy()

        # unit dB turns into unit power
        sensitivity = 10 ** (sensitivity / 20)
        # sensitivity = sensitivity - np.min(sensitivity)

        # normalize sensitivity fall-off function
        sensitivity = sensitivity / np.max(sensitivity)
        sensitivity = sensitivity[self.window[0] : self.window[1]]
        return sensitivity

    def getConfocal(self):
        # read confocal function (actually it reads Rayeligh range)
        fileConfocal = os.path.join(self.confocalPath, "Confocal function.xlsx")
        rayleigh = pd.read_excel(fileConfocal)
        rayleigh = rayleigh["Rayleigh range"][0]
        confocal = self.confocalFunction(
            np.linspace(
                0,
                (self.window[1] - self.window[0]) - 1,
                (self.window[1] - self.window[0]),
            )
            * self.pixelSize,
            rayleigh,
            self.focalPlane * self.pixelSize,
        )
        return confocal

    def confocalFunction(self, x, rayleigh, z0):
        apparentRayleigh = 2 * self.refractionIndex * rayleigh
        temp = ((x - z0) / apparentRayleigh) ** 2
        return (temp + 1) ** (-1)

    def getNoise(self):
        fileNoiseMean = os.path.join(self.intralipidPath, "mean noise power.npy")
        noiseMean = np.load(fileNoiseMean)
        fileNoiseStd = os.path.join(
            self.intralipidPath, "standard deviation of noise power.npy"
        )
        noiseStd = np.load(fileNoiseStd)
        noise = noiseMean * 5
        return (noise, noiseMean)
