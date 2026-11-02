"""Download and convert Places365 pretrained ResNet-18 and ResNet-50 weights for the Context Branch."""

import argparse
import os
import urllib.request
from models.context_branch import convert_places365_checkpoint

PLACES_URLS = {
    "resnet18": "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar",
    "resnet50": "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar",
}


def download_and_prepare(arch: str = "resnet50", store_dir: str = "./places"):
    os.makedirs(store_dir, exist_ok=True)
    tar_path = os.path.join(store_dir, f"{arch}_places365.pth.tar")
    state_dict_path = os.path.join(store_dir, f"{arch}_state_dict.pth")

    if not os.path.exists(tar_path):
        url = PLACES_URLS[arch]
        print(f"Downloading {url} -> {tar_path} ...")
        urllib.request.urlretrieve(url, tar_path)

    print(f"Converting {tar_path} -> {state_dict_path} ...")
    convert_places365_checkpoint(arch=arch, tar_path=tar_path, save_state_dict_path=state_dict_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Places365 ResNet weights")
    parser.add_argument("--arch", type=str, default="resnet50", choices=["resnet18", "resnet50"])
    parser.add_argument("--store_dir", type=str, default="./places")
    args = parser.parse_args()
    download_and_prepare(arch=args.arch, store_dir=args.store_dir)
