"""Tests for ndv.data module functions."""

from pathlib import Path

import numpy as np
import pytest

# ===================== ngff_single_position Tests =====================


def test_ngff_single_position_basic(tmp_path):
    """Test basic creation of single-position OME-Zarr file."""
    yaozarrs = pytest.importorskip("yaozarrs")
    from ndv.data import ngff_single_position

    path = str(tmp_path / "test_single.ome.zarr")
    result_path = ngff_single_position(path)

    assert result_path == path
    assert Path(path).exists()

    # Verify structure
    group = yaozarrs.open_group(path)
    metadata = group.ome_metadata()
    assert hasattr(metadata, "multiscales")


def test_ngff_single_position_custom_shape(tmp_path):
    """Test creating single-position with custom shape."""
    yaozarrs = pytest.importorskip("yaozarrs")
    from ndv.data import ngff_single_position

    path = str(tmp_path / "test_custom_shape.ome.zarr")
    shape = {"t": 5, "c": 3, "z": 2, "y": 128, "x": 128}
    ngff_single_position(path, shape=shape)

    # Verify data shape
    group = yaozarrs.open_group(path)
    arr = group["0"].to_zarr_python()
    assert arr.shape == (5, 3, 2, 128, 128)


def test_ngff_single_position_custom_dtype(tmp_path):
    """Test creating single-position with custom dtype."""
    yaozarrs = pytest.importorskip("yaozarrs")
    from ndv.data import ngff_single_position

    path = str(tmp_path / "test_custom_dtype.ome.zarr")
    ngff_single_position(path, dtype="uint8")

    # Verify dtype
    group = yaozarrs.open_group(path)
    arr = group["0"]
    assert arr.dtype == "uint8"


def test_ngff_single_position_axes(tmp_path):
    """Test that axes are correctly defined (t, c, z, y, x)."""
    yaozarrs = pytest.importorskip("yaozarrs")
    from ndv.data import ngff_single_position

    path = str(tmp_path / "test_axes.ome.zarr")
    ngff_single_position(path)

    group = yaozarrs.open_group(path)
    metadata = group.ome_metadata()
    multiscale = metadata.multiscales[0]

    axis_names = [ax.name for ax in multiscale.axes]
    axis_types = [ax.type for ax in multiscale.axes]

    assert axis_names == ["t", "c", "z", "y", "x"]
    assert axis_types == ["time", "channel", "space", "space", "space"]


def test_ngff_single_position_reproducible_data(tmp_path):
    """Test that data generation is reproducible (uses fixed seed)."""
    from ndv.data import ngff_single_position

    pytest.importorskip("yaozarrs")
    import zarr

    path1 = str(tmp_path / "test1.ome.zarr")
    path2 = str(tmp_path / "test2.ome.zarr")

    shape = {"t": 2, "c": 2, "z": 2, "y": 32, "x": 32}
    ngff_single_position(path1, shape=shape)
    ngff_single_position(path2, shape=shape)

    data1 = zarr.open(path1 + "/0", mode="r")[:]
    data2 = zarr.open(path2 + "/0", mode="r")[:]

    np.testing.assert_array_equal(data1, data2)


def test_ngff_single_position_no_yaozarrs():
    """Test that proper error is raised when yaozarrs is not installed."""
    import sys
    from unittest.mock import patch

    from ndv.data import ngff_single_position

    # Mock ImportError for yaozarrs
    with patch.dict(sys.modules, {"yaozarrs": None, "yaozarrs.write.v05": None}):
        with pytest.raises(ImportError, match="yaozarrs"):
            ngff_single_position("/tmp/test.ome.zarr")


# ===================== ngff_multi_position Tests =====================


def test_ngff_multi_position_basic(tmp_path):
    """Test basic creation of multi-position OME-Zarr file."""
    yaozarrs = pytest.importorskip("yaozarrs")
    from ndv.data import ngff_multi_position

    path = str(tmp_path / "test_multi.ome.zarr")
    result_path = ngff_multi_position(path)

    assert result_path == path
    assert Path(path).exists()

    # Verify root group has bioformats2raw metadata
    group = yaozarrs.open_group(path)
    assert "bioformats2raw.layout" in group.attrs
    assert group.attrs["bioformats2raw.layout"] == 3


def test_ngff_multi_position_positions(tmp_path):
    """Test that correct number of positions are created."""
    yaozarrs = pytest.importorskip("yaozarrs")
    from ndv.data import ngff_multi_position

    path = str(tmp_path / "test_multi_pos.ome.zarr")
    n_positions = 5
    ngff_multi_position(path, n_positions=n_positions)

    # Verify positions exist
    group = yaozarrs.open_group(path)
    for i in range(n_positions):
        subgroup = group[str(i)]
        assert hasattr(subgroup.ome_metadata(), "multiscales")


def test_ngff_multi_position_shape(tmp_path):
    """Test custom shape for each position."""
    yaozarrs = pytest.importorskip("yaozarrs")
    from ndv.data import ngff_multi_position

    path = str(tmp_path / "test_multi_shape.ome.zarr")
    shape = {"t": 3, "c": 2, "y": 64, "x": 64}
    ngff_multi_position(path, n_positions=2, shape=shape)

    # Verify shape
    group = yaozarrs.open_group(path)
    arr = group["0"]["0"].to_zarr_python()
    assert arr.shape == (3, 2, 64, 64)


def test_ngff_multi_position_dtype(tmp_path):
    """Test custom dtype for multi-position."""
    yaozarrs = pytest.importorskip("yaozarrs")
    from ndv.data import ngff_multi_position

    path = str(tmp_path / "test_multi_dtype.ome.zarr")
    ngff_multi_position(path, dtype="uint16")

    # Verify dtype
    group = yaozarrs.open_group(path)
    arr = group["0"]["0"]
    assert arr.dtype == "uint16"


def test_ngff_multi_position_axes(tmp_path):
    """Test that axes are correctly defined for each position."""
    yaozarrs = pytest.importorskip("yaozarrs")
    from ndv.data import ngff_multi_position

    path = str(tmp_path / "test_multi_axes.ome.zarr")
    ngff_multi_position(path)  # Default shape includes t, c, z, y, x

    group = yaozarrs.open_group(path)
    pos_group = group["0"]
    metadata = pos_group.ome_metadata()
    multiscale = metadata.multiscales[0]

    axis_names = [ax.name for ax in multiscale.axes]
    axis_types = [ax.type for ax in multiscale.axes]

    assert axis_names == ["t", "c", "z", "y", "x"]
    assert axis_types == ["time", "channel", "space", "space", "space"]


def test_ngff_multi_position_reproducible_data(tmp_path):
    """Test data reproducibility across positions."""
    pytest.importorskip("yaozarrs")
    import zarr

    from ndv.data import ngff_multi_position

    path1 = str(tmp_path / "test1.ome.zarr")
    path2 = str(tmp_path / "test2.ome.zarr")

    shape = {"t": 2, "c": 2, "y": 32, "x": 32}
    ngff_multi_position(path1, n_positions=2, shape=shape)
    ngff_multi_position(path2, n_positions=2, shape=shape)

    # Verify data in position 0 is the same
    data1_pos0 = zarr.open(path1 + "/0/0", mode="r")[:]
    data2_pos0 = zarr.open(path2 + "/0/0", mode="r")[:]
    np.testing.assert_array_equal(data1_pos0, data2_pos0)


def test_ngff_multi_position_no_yaozarrs():
    """Test that proper error is raised when yaozarrs is not installed."""
    import sys
    from unittest.mock import patch

    from ndv.data import ngff_multi_position

    with patch.dict(
        sys.modules, {"yaozarrs": None, "yaozarrs.write.v05": None, "zarr": None}
    ):
        with pytest.raises(ImportError, match="yaozarrs"):
            ngff_multi_position("/tmp/test.ome.zarr")


# ===================== ngff_plate Tests =====================


def test_ngff_plate_basic(tmp_path):
    """Test basic creation of HCS plate OME-Zarr file."""
    yaozarrs = pytest.importorskip("yaozarrs")
    from ndv.data import ngff_plate

    path = str(tmp_path / "test_plate.ome.zarr")
    result_path = ngff_plate(path)

    assert result_path == path
    assert Path(path).exists()

    # Verify it's a plate
    group = yaozarrs.open_group(path)
    metadata = group.ome_metadata()
    assert hasattr(metadata, "plate")


def test_ngff_plate_dimensions(tmp_path):
    """Test custom plate dimensions (rows x cols)."""
    yaozarrs = pytest.importorskip("yaozarrs")
    from ndv.data import ngff_plate

    path = str(tmp_path / "test_plate_dim.ome.zarr")
    n_rows, n_cols = 3, 4
    ngff_plate(path, n_rows=n_rows, n_cols=n_cols, n_fovs=1)

    # Verify plate metadata
    group = yaozarrs.open_group(path)
    metadata = group.ome_metadata()
    plate_def = metadata.plate

    assert len(plate_def.rows) == n_rows
    assert len(plate_def.columns) == n_cols
    assert len(plate_def.wells) == n_rows * n_cols


def test_ngff_plate_fovs(tmp_path):
    """Test multiple FOVs per well."""
    yaozarrs = pytest.importorskip("yaozarrs")
    from ndv.data import ngff_plate

    path = str(tmp_path / "test_plate_fovs.ome.zarr")
    n_fovs = 3
    ngff_plate(path, n_rows=2, n_cols=2, n_fovs=n_fovs)

    # Verify FOV count in metadata
    group = yaozarrs.open_group(path)
    metadata = group.ome_metadata()
    assert metadata.plate.field_count == n_fovs

    # Verify individual well has multiple FOVs
    well_group = group["A"]["1"]
    # Should have 3 FOVs numbered 0, 1, 2
    for i in range(n_fovs):
        # Check if FOV exists
        fov = well_group[str(i)]
        assert fov is not None


def test_ngff_plate_well_structure(tmp_path):
    """Test well structure (row/col naming)."""
    yaozarrs = pytest.importorskip("yaozarrs")
    from ndv.data import ngff_plate

    path = str(tmp_path / "test_plate_wells.ome.zarr")
    ngff_plate(path, n_rows=2, n_cols=3, n_fovs=1)

    group = yaozarrs.open_group(path)

    # Check row names (A, B) and column names (1, 2, 3)
    for row in ["A", "B"]:
        row_group = group[row]
        assert row_group is not None
        for col in ["1", "2", "3"]:
            col_group = row_group[col]
            assert col_group is not None


def test_ngff_plate_shape(tmp_path):
    """Test custom FOV shape."""
    yaozarrs = pytest.importorskip("yaozarrs")
    from ndv.data import ngff_plate

    path = str(tmp_path / "test_plate_shape.ome.zarr")
    shape = {"c": 3, "z": 4, "y": 128, "x": 128}
    ngff_plate(path, n_rows=1, n_cols=1, n_fovs=1, shape=shape)

    # Verify shape
    group = yaozarrs.open_group(path)
    arr = group["A"]["1"]["0"]["0"].to_zarr_python()
    assert arr.shape == (3, 4, 128, 128)


def test_ngff_plate_dtype(tmp_path):
    """Test custom dtype for plate."""
    yaozarrs = pytest.importorskip("yaozarrs")
    from ndv.data import ngff_plate

    path = str(tmp_path / "test_plate_dtype.ome.zarr")
    ngff_plate(path, dtype="uint8")

    # Verify dtype
    group = yaozarrs.open_group(path)
    arr = group["A"]["1"]["0"]["0"]
    assert arr.dtype == "uint8"


def test_ngff_plate_axes(tmp_path):
    """Test that axes are correctly defined for each FOV."""
    yaozarrs = pytest.importorskip("yaozarrs")
    from ndv.data import ngff_plate

    path = str(tmp_path / "test_plate_axes.ome.zarr")
    ngff_plate(path)  # Default shape includes t, c, z, y, x

    group = yaozarrs.open_group(path)
    fov_group = group["A"]["1"]["0"]
    metadata = fov_group.ome_metadata()
    multiscale = metadata.multiscales[0]

    axis_names = [ax.name for ax in multiscale.axes]
    axis_types = [ax.type for ax in multiscale.axes]

    assert axis_names == ["t", "c", "z", "y", "x"]
    assert axis_types == ["time", "channel", "space", "space", "space"]


def test_ngff_plate_reproducible_data(tmp_path):
    """Test data reproducibility across plate creations."""
    pytest.importorskip("yaozarrs")
    import zarr

    from ndv.data import ngff_plate

    path1 = str(tmp_path / "test1.ome.zarr")
    path2 = str(tmp_path / "test2.ome.zarr")

    shape = {"c": 2, "z": 2, "y": 32, "x": 32}
    ngff_plate(path1, n_rows=2, n_cols=2, n_fovs=1, shape=shape)
    ngff_plate(path2, n_rows=2, n_cols=2, n_fovs=1, shape=shape)

    # Verify data in well A/1/0 is the same
    data1 = zarr.open(path1 + "/A/1/0/0", mode="r")[:]
    data2 = zarr.open(path2 + "/A/1/0/0", mode="r")[:]
    np.testing.assert_array_equal(data1, data2)


def test_ngff_plate_no_yaozarrs():
    """Test that proper error is raised when yaozarrs is not installed."""
    import sys
    from unittest.mock import patch

    from ndv.data import ngff_plate

    with patch.dict(sys.modules, {"yaozarrs": None, "yaozarrs.write.v05": None}):
        with pytest.raises(ImportError, match="yaozarrs"):
            ngff_plate("/tmp/test.ome.zarr")


def test_ngff_plate_row_naming(tmp_path):
    """Test that rows are named correctly (A, B, C, ...)."""
    yaozarrs = pytest.importorskip("yaozarrs")
    from ndv.data import ngff_plate

    path = str(tmp_path / "test_plate_rows.ome.zarr")
    n_rows = 5
    ngff_plate(path, n_rows=n_rows, n_cols=1, n_fovs=1)

    group = yaozarrs.open_group(path)
    metadata = group.ome_metadata()

    row_names = [row.name for row in metadata.plate.rows]
    expected_names = ["A", "B", "C", "D", "E"]
    assert row_names == expected_names


def test_ngff_plate_col_naming(tmp_path):
    """Test that columns are named correctly (1, 2, 3, ...) 1-indexed."""
    yaozarrs = pytest.importorskip("yaozarrs")
    from ndv.data import ngff_plate

    path = str(tmp_path / "test_plate_cols.ome.zarr")
    n_cols = 4
    ngff_plate(path, n_rows=1, n_cols=n_cols, n_fovs=1)

    group = yaozarrs.open_group(path)
    metadata = group.ome_metadata()

    col_names = [col.name for col in metadata.plate.columns]
    expected_names = ["1", "2", "3", "4"]
    assert col_names == expected_names


# ===================== Integration Tests =====================


def test_all_ngff_functions_in_all(tmp_path):
    """Test that all NGFF functions are exported in __all__."""
    import ndv.data

    assert "ngff_single_position" in ndv.data.__all__
    assert "ngff_multi_position" in ndv.data.__all__
    assert "ngff_plate" in ndv.data.__all__


def test_ngff_functions_accessible_from_ndv(tmp_path):
    """Test that NGFF functions can be accessed from ndv.data."""
    import ndv

    assert hasattr(ndv.data, "ngff_single_position")
    assert hasattr(ndv.data, "ngff_multi_position")
    assert hasattr(ndv.data, "ngff_plate")


def test_ngff_overwrite_behavior(tmp_path):
    """Test that overwrite=True allows recreating files."""
    pytest.importorskip("yaozarrs")
    from ndv.data import ngff_single_position

    path = str(tmp_path / "test_overwrite.ome.zarr")

    # Create first file
    ngff_single_position(path, shape={"t": 2, "c": 2, "z": 2, "y": 32, "x": 32})

    # Create again (should succeed due to overwrite=True)
    ngff_single_position(path, shape={"t": 3, "c": 3, "z": 3, "y": 64, "x": 64})

    # Verify new shape
    import zarr

    data = zarr.open(path + "/0", mode="r")
    assert data.shape == (3, 3, 3, 64, 64)
