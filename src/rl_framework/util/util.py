import d3rlpy.torch_utility
import datasets.filesystems
import torch as th
from imitation.util import util


# Monkey-patch the function since it fails to detect local file as tuple protocol
def patch_datasets():
    modules = [datasets.arrow_dataset.__dict__, datasets.filesystems.__dict__, datasets.builder.__dict__]
    old = datasets.filesystems.is_remote_filesystem

    def custom_is_remote_filesystem(fs):
        return old(fs) and fs.protocol != ("file", "local")

    for module in modules:
        module["is_remote_filesystem"] = custom_is_remote_filesystem


# Fixed in d3rlpy 2.8.0, but that version is not compatible with other dependencies
def patch_d3rlpy():
    def map_location(device: str):
        if "cuda" in device:
            if ":" in device:
                _, index = device.split(":")
            else:
                index = "0"
            return lambda storage, loc: storage.cuda(int(index))

        if "cpu" in device:
            return "cpu"

        raise ValueError(f"invalid device={device}")

    d3rlpy.torch_utility.map_location = map_location


def patch_imitation_safe_to_tensor():
    def patched_safe_to_tensor(array, **kwargs):
        array = old_safe_to_tensor(array, **kwargs)
        return th.as_tensor(array, **kwargs)

    old_safe_to_tensor = util.safe_to_tensor
    util.safe_to_tensor = patched_safe_to_tensor
