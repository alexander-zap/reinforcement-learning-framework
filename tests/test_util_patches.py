"""Monkey patches applied to third-party libraries (datasets, d3rlpy, imitation)."""

from types import SimpleNamespace

import d3rlpy.torch_utility
import datasets.filesystems
import numpy as np
import pytest
import torch as th
from imitation.util import util as imitation_util

from rl_framework.util import (
    patch_d3rlpy,
    patch_datasets,
    patch_imitation_safe_to_tensor,
)


def test_patched_datasets_treat_local_protocol_tuple_as_local():
    patch_datasets()

    is_remote = datasets.filesystems.is_remote_filesystem
    assert is_remote(SimpleNamespace(protocol=("file", "local"))) is False
    assert datasets.arrow_dataset.is_remote_filesystem is is_remote
    assert datasets.builder.is_remote_filesystem is is_remote


@pytest.mark.parametrize("device", ["cpu", "cpu:0"])
def test_patched_d3rlpy_maps_cpu_devices_to_cpu(device):
    patch_d3rlpy()
    assert d3rlpy.torch_utility.map_location(device) == "cpu"


@pytest.mark.parametrize("device", ["cuda", "cuda:1"])
def test_patched_d3rlpy_maps_cuda_devices_to_a_storage_mover(device):
    patch_d3rlpy()
    assert callable(d3rlpy.torch_utility.map_location(device))


def test_patched_d3rlpy_rejects_unknown_devices():
    patch_d3rlpy()
    with pytest.raises(ValueError, match="invalid device"):
        d3rlpy.torch_utility.map_location("tpu")


def test_patched_imitation_safe_to_tensor_returns_tensors():
    patch_imitation_safe_to_tensor()

    tensor = imitation_util.safe_to_tensor(np.arange(3, dtype=np.float32), device="cpu")

    assert isinstance(tensor, th.Tensor)
    th.testing.assert_close(tensor, th.arange(3, dtype=th.float32))
