import datasets.filesystems


# Monkey-patch the function since it fails to detect local file as tuple protocol
def patch_datasets():
    module = datasets.arrow_dataset.__dict__
    old = module["is_remote_filesystem"]

    def custom_is_remote_filesystem(fs):
        return old(fs) and fs.protocol != ("file", "local")

    module["is_remote_filesystem"] = custom_is_remote_filesystem
