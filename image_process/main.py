from utils.image_reconstruction import ReconstructImage
from utils.despeckle import despeckle
from utils.resize_and_normalize import ResizeAndNormalize
import json


if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)

    print(config)

    # image reconstruction
    reconstruct = ReconstructImage(config["process"]["reconstructImage"])
    reconstruct.reconstruct()

    # image denoise
    denoise = despeckle(config["process"]["despeckle"])
    denoise.denoise()

    # resize and normalize
    preprocess = ResizeAndNormalize(config["process"]["preprocess"])
    preprocess.preprocess()
