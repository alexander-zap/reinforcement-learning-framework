import tempfile

import matplotlib
import pytest

# Headless plotting for connector histogram tests.
matplotlib.use("Agg")


@pytest.fixture(scope="session", autouse=True)
def isolated_tempdir(tmp_path_factory):
    """Agents create a tensorboard temp dir per instance; keep those (and other temp files) inside pytest's tmp dir."""
    previous = tempfile.tempdir
    tempfile.tempdir = str(tmp_path_factory.mktemp("tempfile"))
    yield
    tempfile.tempdir = previous
