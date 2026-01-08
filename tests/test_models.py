from unittest.mock import Mock

import pytest

from ndv.models._array_display_model import ArrayDisplayModel
from ndv.models._roi_model import RectangularROIModel


def test_array_display_model() -> None:
    m = ArrayDisplayModel()

    mock = Mock()
    m.events.channel_axis.connect(mock)
    m.current_index.item_added.connect(mock)
    m.current_index.item_changed.connect(mock)

    m.channel_axis = 4
    mock.assert_called_once_with(4, None)  # new, old
    mock.reset_mock()
    m.current_index["5"] = 1
    mock.assert_called_once_with(5, 1)  # key, value
    mock.reset_mock()
    m.current_index[5] = 4
    mock.assert_called_once_with(5, 4, 1)  # key, new, old
    mock.reset_mock()

    assert ArrayDisplayModel.model_json_schema(mode="validation")
    assert ArrayDisplayModel.model_json_schema(mode="serialization")


def test_rectangular_roi_model() -> None:
    m = RectangularROIModel()

    mock = Mock()
    m.events.bounding_box.connect(mock)
    m.events.visible.connect(mock)

    m.bounding_box = ((10, 10), (20, 20))
    mock.assert_called_once_with(
        ((10, 10), (20, 20)),  # New bounding box value
        ((0, 0), (0, 0)),  # Initial bounding box on construction
    )
    mock.reset_mock()

    m.visible = False
    mock.assert_called_once_with(
        False,  # New visibility
        True,  # Initial visibility on construction
    )
    mock.reset_mock()

    assert RectangularROIModel.model_json_schema(mode="validation")
    assert RectangularROIModel.model_json_schema(mode="serialization")


# ===================== NGFF Wrapper Tests =====================


@pytest.fixture
def temp_ngff_files(tmp_path):
    """Create temporary NGFF test files."""
    import ndv

    files = {}

    # Single position
    sp_path = str(tmp_path / "test_single.ome.zarr")
    ndv.data.ngff_single_position(sp_path, shape=(2, 2, 3, 32, 32))
    files["single"] = sp_path

    # Multi-position
    mp_path = str(tmp_path / "test_multipos.ome.zarr")
    ndv.data.ngff_multi_position(mp_path, n_positions=4, shape=(2, 2, 32, 32))
    files["multi"] = mp_path

    # Plate
    pl_path = str(tmp_path / "test_plate.ome.zarr")
    ndv.data.ngff_plate(pl_path, n_rows=2, n_cols=3, n_fovs=2)
    files["plate"] = pl_path

    return files


def test_ngff_wrapper_single_position(temp_ngff_files):
    """Test NGFFWrapper with single-position OME-Zarr."""
    pytest.importorskip("yaozarrs")
    from ndv.models._ngff_wrapper import NGFFWrapper

    wrapper = NGFFWrapper(temp_ngff_files["single"])

    # Check dimensions
    assert wrapper.dims == ("t", "c", "z", "y", "x")
    assert wrapper.sizes() == {"t": 2, "c": 2, "z": 3, "y": 32, "x": 32}
    assert wrapper.dtype.name == "uint16"

    # Check channel axis detection
    assert wrapper.guess_channel_axis() == 1

    # Test slicing with slice objects (what ndv does internally)
    data = wrapper.isel(
        {
            0: slice(0, 1),  # t
            1: slice(0, 1),  # c
            2: slice(None),  # z
            3: slice(None),  # y
            4: slice(None),  # x
        }
    )
    assert data.shape == (1, 1, 3, 32, 32)
    assert data.dtype.name == "uint16"


def test_ngff_wrapper_multi_position(temp_ngff_files):
    """Test NGFFWrapper with multi-position OME-Zarr."""
    pytest.importorskip("yaozarrs")
    from ndv.models._ngff_wrapper import NGFFWrapper

    wrapper = NGFFWrapper(temp_ngff_files["multi"])

    # Check dimensions - should have 'p' as first dimension
    assert wrapper.dims == ("p", "t", "c", "y", "x")
    assert wrapper.sizes() == {"p": 4, "t": 2, "c": 2, "y": 32, "x": 32}
    assert wrapper.dtype.name == "uint8"

    # Check channel axis detection
    assert wrapper.guess_channel_axis() == 2

    # Test slicing - slice should preserve singleton dimensions
    data = wrapper.isel(
        {
            0: slice(0, 1),  # p
            1: slice(0, 1),  # t
            2: slice(None),  # c
            3: slice(None),  # y
            4: slice(None),  # x
        }
    )
    assert data.shape == (1, 1, 2, 32, 32)
    assert data.dtype.name == "uint8"

    # Test different position
    data = wrapper.isel(
        {
            0: slice(2, 3),  # p=2
            1: slice(0, 1),  # t
            2: slice(None),  # c
            3: slice(None),  # y
            4: slice(None),  # x
        }
    )
    assert data.shape == (1, 1, 2, 32, 32)


def test_ngff_wrapper_plate(temp_ngff_files):
    """Test NGFFWrapper with HCS plate OME-Zarr."""
    pytest.importorskip("yaozarrs")
    from ndv.models._ngff_wrapper import NGFFWrapper

    wrapper = NGFFWrapper(temp_ngff_files["plate"])

    # Check dimensions - should have 'p' as first dimension
    # 2 rows x 3 cols x 2 FOVs = 12 positions
    assert wrapper.dims == ("p", "c", "z", "y", "x")
    assert wrapper.sizes() == {"p": 12, "c": 2, "z": 3, "y": 64, "x": 64}
    assert wrapper.dtype.name == "uint16"

    # Check channel axis detection
    assert wrapper.guess_channel_axis() == 1

    # Test slicing position 0
    data = wrapper.isel(
        {
            0: slice(0, 1),  # p
            1: slice(0, 1),  # c
            2: slice(None),  # z
            3: slice(None),  # y
            4: slice(None),  # x
        }
    )
    assert data.shape == (1, 1, 3, 64, 64)

    # Test slicing position 5
    data = wrapper.isel(
        {
            0: slice(5, 6),  # p
            1: slice(0, 1),  # c
            2: slice(None),  # z
            3: slice(None),  # y
            4: slice(None),  # x
        }
    )
    assert data.shape == (1, 1, 3, 64, 64)


def test_ngff_wrapper_supports():
    """Test NGFFWrapper.supports() method."""
    pytest.importorskip("yaozarrs")
    from ndv.models._ngff_wrapper import NGFFWrapper

    # Should support .ome.zarr paths
    assert NGFFWrapper.supports("/path/to/file.ome.zarr")

    # Should not support other paths
    assert not NGFFWrapper.supports("/path/to/file.zarr")
    assert not NGFFWrapper.supports("/path/to/file.tif")

    # Should not support non-string types
    assert not NGFFWrapper.supports(123)
    assert not NGFFWrapper.supports([1, 2, 3])


def test_ngff_wrapper_with_data_wrapper_create(temp_ngff_files):
    """Test that DataWrapper.create() properly detects NGFF files."""
    pytest.importorskip("yaozarrs")
    from ndv.models._data_wrapper import DataWrapper
    from ndv.models._ngff_wrapper import NGFFWrapper

    # Test single position
    wrapper = DataWrapper.create(temp_ngff_files["single"])
    assert isinstance(wrapper, NGFFWrapper)
    assert wrapper.dims == ("t", "c", "z", "y", "x")

    # Test multi-position
    wrapper = DataWrapper.create(temp_ngff_files["multi"])
    assert isinstance(wrapper, NGFFWrapper)
    assert wrapper.dims == ("p", "t", "c", "y", "x")

    # Test plate
    wrapper = DataWrapper.create(temp_ngff_files["plate"])
    assert isinstance(wrapper, NGFFWrapper)
    assert wrapper.dims == ("p", "c", "z", "y", "x")
