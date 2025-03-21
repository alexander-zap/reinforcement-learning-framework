import d3rlpy.torch_utility
import datasets.filesystems


# Monkey-patch the function since it fails to detect local file as tuple protocol
def patch_datasets():
    module = datasets.arrow_dataset.__dict__
    old = module["is_remote_filesystem"]

    def custom_is_remote_filesystem(fs):
        return old(fs) and fs.protocol != ("file", "local")

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
