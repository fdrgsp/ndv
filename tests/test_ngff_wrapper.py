import pytest

from ndv.models._ngff_wrapper import NGFFWrapper

from ._utils import ngff_multi_position, ngff_plate, ngff_single_position

pytest.importorskip("yaozarrs", reason="yaozarrs is required for NGFFWrapper tests")


@pytest.fixture(scope="module")
def single_position(tmp_path_factory):
    """Create a single-position OME-Zarr file."""
    path = tmp_path_factory.mktemp("ngff") / "single.ome.zarr"
    return ngff_single_position(
        str(path), shape={"t": 2, "c": 2, "z": 4, "y": 32, "x": 32}
    )


@pytest.fixture(scope="module")
def multi_position(tmp_path_factory):
    """Create a multi-position OME-Zarr file."""
    path = tmp_path_factory.mktemp("ngff") / "multi.ome.zarr"
    return ngff_multi_position(
        str(path), n_positions=3, shape={"t": 2, "c": 2, "z": 4, "y": 32, "x": 32}
    )


@pytest.fixture(scope="module")
def plate(tmp_path_factory):
    """Create a plate OME-Zarr file."""
    path = tmp_path_factory.mktemp("ngff") / "plate.ome.zarr"
    return ngff_plate(
        str(path),
        n_rows=2,
        n_cols=2,
        n_fovs=2,
        shape={"t": 2, "c": 2, "z": 4, "y": 32, "x": 32},
    )


def test_ngff_wrapper_single_position(single_position):
    """Test NGFFWrapper with single-position OME-Zarr."""
    wrapper = NGFFWrapper(single_position)

    # Check dimensions
    assert wrapper.dims == ("t", "c", "z", "y", "x")
    assert wrapper.sizes() == {"t": 2, "c": 2, "z": 4, "y": 32, "x": 32}
    assert wrapper.dtype.name == "uint16"

    # # Check channel axis detection
    assert wrapper.guess_channel_axis() == 1

    # # Test slicing with slice objects (what ndv does internally)
    data = wrapper.isel(
        {
            0: slice(0, 1),  # t
            1: slice(0, 1),  # c
            2: slice(0, 1),  # z
            3: slice(None),  # y
            4: slice(None),  # x
        }
    )
    assert data.shape == (1, 1, 1, 32, 32)
    assert data.dtype.name == "uint16"


def test_ngff_wrapper_multi_position(multi_position):
    """Test NGFFWrapper with multi-position OME-Zarr."""
    wrapper = NGFFWrapper(multi_position)

    # Check dimensions - should have 'p' as first dimension
    assert wrapper.dims == ("p", "t", "c", "z", "y", "x")
    assert wrapper.sizes() == {"p": 3, "t": 2, "c": 2, "z": 4, "y": 32, "x": 32}
    assert wrapper.dtype.name == "uint8"

    # Check channel axis detection
    assert wrapper.guess_channel_axis() == 2

    # Test slicing - slice should preserve singleton dimensions
    data = wrapper.isel(
        {
            0: slice(0, 1),  # p
            1: slice(None),  # t
            2: slice(None),  # c
            3: slice(None),  # z
            4: slice(None),  # y
            5: slice(None),  # x
        }
    )
    assert data.shape == (1, 2, 2, 4, 32, 32)
    assert data.dtype.name == "uint8"

    # Test different position
    data = wrapper.isel(
        {
            0: slice(2, 3),  # p=2
            1: slice(None),  # t
            2: slice(None),  # c
            3: slice(None),  # z
            4: slice(None),  # y
            5: slice(None),  # x
        }
    )
    assert data.shape == (1, 2, 2, 4, 32, 32)


def test_ngff_wrapper_plate(plate):
    """Test NGFFWrapper with HCS plate OME-Zarr."""
    wrapper = NGFFWrapper(plate)

    # Check dimensions - should have 'p' as first dimension
    # Plate with 2x2 wells, 2 FOVs each = 8 positions
    assert wrapper.dims == ("p", "t", "c", "z", "y", "x")
    assert wrapper.sizes() == {"p": 8, "t": 2, "c": 2, "z": 4, "y": 32, "x": 32}
    assert wrapper.dtype.name == "uint16"

    # Check channel axis detection
    assert wrapper.guess_channel_axis() == 2

    # Test slicing position 0
    data = wrapper.isel(
        {
            0: slice(0, 1),  # p
            1: slice(0, 1),  # t
            2: slice(0, 1),  # c
            3: slice(None),  # z
            4: slice(None),  # y
            5: slice(None),  # x
        }
    )
    assert data.shape == (1, 1, 1, 4, 32, 32)

    # Test slicing position 5
    data = wrapper.isel(
        {
            0: slice(5, 6),  # p
            1: slice(0, 1),  # t
            2: slice(0, 1),  # c
            3: slice(None),  # z
            4: slice(None),  # y
            5: slice(None),  # x
        }
    )
    assert data.shape == (1, 1, 1, 4, 32, 32)


def test_ngff_wrapper_supports():
    """Test NGFFWrapper.supports() method."""

    # Should support .ome.zarr paths
    assert NGFFWrapper.supports("/path/to/file.ome.zarr")

    # Should not support other paths
    assert not NGFFWrapper.supports("/path/to/file.zarr")
    assert not NGFFWrapper.supports("/path/to/file.tif")

    # Should not support non-string types
    assert not NGFFWrapper.supports(123)
    assert not NGFFWrapper.supports([1, 2, 3])


def test_ngff_wrapper_with_data_wrapper_create(single_position, multi_position, plate):
    """Test that DataWrapper.create() properly detects NGFF files."""
    from ndv.models._data_wrapper import DataWrapper

    # Test single position
    wrapper = DataWrapper.create(single_position)
    assert isinstance(wrapper, NGFFWrapper)
    assert wrapper.dims == ("t", "c", "z", "y", "x")

    # Test multi-position
    wrapper = DataWrapper.create(multi_position)
    assert isinstance(wrapper, NGFFWrapper)
    assert wrapper.dims == ("p", "t", "c", "z", "y", "x")

    # Test plate
    wrapper = DataWrapper.create(plate)
    assert isinstance(wrapper, NGFFWrapper)
    assert wrapper.dims == ("p", "t", "c", "z", "y", "x")
