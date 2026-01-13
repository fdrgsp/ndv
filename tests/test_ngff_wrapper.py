import pytest

from ndv.models._data_wrapper import DataWrapper, NGFFWrapper

from ._utils import ngff_multi_position, ngff_plate, ngff_single_position

pytest.importorskip("yaozarrs")

SHAPE = {"t": 2, "c": 2, "z": 4, "y": 32, "x": 32}


@pytest.fixture(scope="module")
def single_position(tmp_path_factory):
    path = tmp_path_factory.mktemp("ngff") / "single.ome.zarr"
    return ngff_single_position(str(path), shape=SHAPE)


@pytest.fixture(scope="module")
def multi_position(tmp_path_factory):
    path = tmp_path_factory.mktemp("ngff") / "multi.ome.zarr"
    return ngff_multi_position(str(path), n_positions=3, shape=SHAPE)


@pytest.fixture(scope="module")
def plate(tmp_path_factory):
    path = tmp_path_factory.mktemp("ngff") / "plate.ome.zarr"
    return ngff_plate(str(path), n_rows=2, n_cols=2, n_fovs=2, shape=SHAPE)


def test_single_position(single_position):
    wrapper = NGFFWrapper(single_position)
    assert wrapper.dims == ("t", "c", "z", "y", "x")
    assert wrapper.sizes() == {"t": 2, "c": 2, "z": 4, "y": 32, "x": 32}
    assert wrapper.dtype.name == "uint8"
    assert wrapper.guess_channel_axis() == 1
    data = wrapper.isel({0: slice(0, 1), 1: slice(0, 1), 2: slice(0, 1)})
    assert data.shape == (1, 1, 1, 32, 32)


def test_multi_position(multi_position):
    wrapper = NGFFWrapper(multi_position)
    assert wrapper.dims == ("p", "t", "c", "z", "y", "x")
    assert wrapper.sizes() == {"p": 3, "t": 2, "c": 2, "z": 4, "y": 32, "x": 32}
    assert wrapper.dtype.name == "uint8"
    assert wrapper.guess_channel_axis() == 2
    data = wrapper.isel({0: slice(0, 1)})
    assert data.shape == (1, 2, 2, 4, 32, 32)


def test_plate(plate):
    wrapper = NGFFWrapper(plate)
    assert wrapper.dims == ("p", "t", "c", "z", "y", "x")
    assert wrapper.sizes() == {"p": 8, "t": 2, "c": 2, "z": 4, "y": 32, "x": 32}
    assert wrapper.dtype.name == "uint8"
    assert wrapper.guess_channel_axis() == 2
    data = wrapper.isel({0: slice(0, 1), 1: slice(0, 1), 2: slice(0, 1)})
    assert data.shape == (1, 1, 1, 4, 32, 32)


def test_supports(single_position):
    # Only returns True for paths that actually exist and have valid OME metadata
    assert NGFFWrapper.supports(single_position)
    assert not NGFFWrapper.supports("/path/to/nonexistent.ome.zarr")
    assert not NGFFWrapper.supports("/path/to/file.zarr")
    assert not NGFFWrapper.supports(123)


def test_data_wrapper_create(single_position, multi_position, plate):
    assert isinstance(DataWrapper.create(single_position), NGFFWrapper)
    assert isinstance(DataWrapper.create(multi_position), NGFFWrapper)
    assert isinstance(DataWrapper.create(plate), NGFFWrapper)
