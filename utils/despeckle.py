import os
import numpy as np
import copy
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.interpolate import UnivariateSpline

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""

    Speckle Reduction (去除光斑)
    
    filePath: 要處理影像的資料夾路徑
    
    dirNameToSolve: filePath 當中要處理的資料夾名稱
    
    舉例來說，filePath 中包含了各個組織的影像資料夾，而包含要處理光斑的影像的資料夾名稱就是
    dirNameToSolve
    
    numOfAverage: 選擇要平均的影像數，以 5 為例，會拿出要處理光斑的影像前後各 5 張影像，
                    因此總共會平均 11 張影像

""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""


class despeckle:
    def __init__(self, config):
        self.filePath = config["filePath"]
        self.dirNameToSolve = config["dirNameToSolve"]
        self.numOfAverage = config["numOfAverage"]

    def denoise(self):
        for dirPath, dirNames, fileNames in os.walk(self.filePath):
            if self.dirNameToSolve in dirNames:
                pathDenoise = os.path.join(dirPath, "denoise")

                if not os.path.exists(pathDenoise):
                    os.mkdir(pathDenoise)

            imagePowerStack = []
            indexOfImagePowerStack = []

            if os.path.basename(dirPath) == self.dirNameToSolve:
                os.chdir(pathDenoise)

                for file in fileNames:
                    if file.endswith(".npy"):
                        imageDB = np.load(os.path.join(dirPath, file))

                        """""" """""" """""" """""" """""" """""" """""
                            Remove straight artifact
                            Check if straight artifact exists
                        
                        """ """""" """""" """""" """""" """""" """""" ""

                        indexOfStraightArtifact = []
                        for x in range(imageDB.shape[1]):
                            if np.where(imageDB[:, x] > 85)[0].shape[0] >= 250:
                                # find straight artifact
                                indexOfStraightArtifact.append(x)

                        straight_artifact_index_temp = copy.copy(
                            indexOfStraightArtifact
                        )

                        for x in indexOfStraightArtifact:
                            # Getting surface index by interpolation
                            pixel_range = [0, 0]

                            pixel_range[0] = max(x - 50, 0)
                            pixel_range[1] = min(x + 50, imageDB.shape[1] - 1)

                            tempX = np.linspace(
                                pixel_range[0],
                                pixel_range[1],
                                pixel_range[1] - pixel_range[0] + 1,
                            )

                            index_to_remove = []
                            for index in range(pixel_range[1] - pixel_range[0] + 1):
                                if tempX[index] in straight_artifact_index_temp:
                                    index_to_remove.append(index)

                            X = np.delete(tempX, index_to_remove)

                            for z in range(imageDB.shape[0]):
                                tempY = imageDB[z, pixel_range[0] : pixel_range[1] + 1]
                                Y = np.delete(tempY, index_to_remove)
                                spline = UnivariateSpline(X, Y, k=1)

                                if spline(np.array(x)) < 0:
                                    imageDB[z, x] = 0

                                else:
                                    imageDB[z, x] = spline(np.array(x))

                            straight_artifact_index_temp.remove(x)

                        # unit dB turns into unit power
                        imagePowerStack.append(10 ** (imageDB / 20))

                        indexOfImagePowerStack.append(int(file.split("_")[0]))

                zipped = zip(indexOfImagePowerStack, imagePowerStack)
                zipped = sorted(zipped, key=lambda x: x[0])
                indexOfImagePowerStack, imagePowerStack = zip(*zipped)

                for num in range(len(imagePowerStack)):
                    each_side_average = [
                        max(0, num - self.numOfAverage),
                        min(len(imagePowerStack) - 1, num + self.numOfAverage),
                    ]

                    average_Bscan = np.zeros_like(imagePowerStack[0])
                    center_image = copy.copy(imagePowerStack[num])

                    fixed = sitk.GetImageFromArray(center_image)
                    for index in range(each_side_average[0], each_side_average[1] + 1):

                        moving = sitk.GetImageFromArray(imagePowerStack[index])

                        """""" """""" """""" """""" """""" """                
                            Translation Registration
                        
                        """ """""" """""" """""" """""" """"""

                        registration = sitk.ImageRegistrationMethod()
                        registration.SetMetricAsMattesMutualInformation(24)
                        registration.SetMetricSamplingPercentage(
                            0.1, sitk.sitkWallClock
                        )
                        registration.SetMetricSamplingStrategy(registration.RANDOM)
                        registration.SetOptimizerAsRegularStepGradientDescent(
                            learningRate=1.0, minStep=0.001, numberOfIterations=150
                        )
                        registration.SetInitialTransform(
                            sitk.TranslationTransform(fixed.GetDimension())
                        )
                        registration.SetInterpolator(sitk.sitkLinear)

                        # only integers could be passed
                        registration.SetShrinkFactorsPerLevel([9, 1, 1])

                        registration.SetSmoothingSigmasPerLevel([1, 2, 1])

                        outTx = registration.Execute(fixed, moving)

                        resampler = sitk.ResampleImageFilter()
                        resampler.SetReferenceImage(fixed)
                        resampler.SetInterpolator(sitk.sitkBSplineResamplerOrder3)
                        resampler.SetDefaultPixelValue(0)
                        resampler.SetTransform(outTx)
                        resultImage = resampler.Execute(moving)

                        resultImage = sitk.GetArrayFromImage(resultImage)

                        average_Bscan = average_Bscan + resultImage

                    average_Bscan = average_Bscan / (
                        each_side_average[1] - each_side_average[0] + 1
                    )
                    average_Bscan[average_Bscan <= 0] = 0.001
                    average_Bscan = 20 * np.log10(average_Bscan)

                    # save
                    plt.imsave(
                        "{}_image.png".format(num + 1),
                        average_Bscan,
                        cmap="gray",
                        vmin=65,
                        vmax=120,
                    )  # OCT: [65,120]

                    np.save(
                        "{}_image".format(num + 1), average_Bscan, allow_pickle=False
                    )
