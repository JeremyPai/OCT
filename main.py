from utils.image_reconstruction import ReconstructImage
import json


if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)

    print(config)

    reconstruct = ReconstructImage(config["process"]["reconstructImage"])
    reconstruct.reconstruct()
