from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

# Canonical axis order for NGFF data
_AXIS_ORDER = ("t", "c", "z", "y", "x")

# Default shapes for each function
DEFAULT_SINGLE_SHAPE: dict[str, int] = {"t": 3, "c": 2, "z": 3, "y": 32, "x": 32}


def ngff_single_position(
    path: str,
    shape: Mapping[str, int] = DEFAULT_SINGLE_SHAPE,
    dtype: str = "uint16",
) -> str:
    """Create a single-position OME-NGFF (OME-Zarr) file.

    Creates a standard multiscale OME-Zarr image. Requires `yaozarrs[write]`.

    Parameters
    ----------
    path : str
        Path where the .ome.zarr directory will be created.
    shape : Mapping[str, int], optional
        Mapping of axis names to sizes. Must include "y" and "x".
        Optional axes: "t" (time), "c" (channel), "z" (z-slices).
        Default: {"t": 3, "c": 2, "z": 3, "y": 32, "x": 32}
    dtype : str, optional
        Data type for the array. Default is "uint16".

    Returns
    -------
    str
        Path to the created OME-Zarr store.

    Examples
    --------
    >>> import ndv
    >>> # Full 5D dataset
    >>> path = ndv.data.ngff_single_position("/tmp/test.ome.zarr")
    >>> # 3D dataset (z, y, x only)
    >>> path = ndv.data.ngff_single_position(
    ...     "/tmp/test3d.ome.zarr", shape={"z": 10, "y": 128, "x": 128}
    ... )
    """
    try:
        from yaozarrs import v05
        from yaozarrs.write.v05 import write_image
    except ImportError as e:
        raise ImportError(
            "Please install yaozarrs with write support: pip install 'yaozarrs[write]'"
        ) from e

    shape_tuple = _shape_to_tuple(shape)
    axes = _build_axes(shape)

    rng = np.random.default_rng(42)
    data = rng.integers(0, np.iinfo(dtype).max // 4, size=shape_tuple, dtype=dtype)

    # Build chunks: 1 for t/c, up to 64 for spatial axes
    chunks = tuple(
        1 if ax in ("t", "c") else min(shape[ax], 64)
        for ax in _AXIS_ORDER
        if ax in shape
    )

    image = v05.Image(
        multiscales=[
            v05.Multiscale(
                name="single_position_example",
                axes=axes,
                datasets=[
                    v05.Dataset(
                        path="0",
                        coordinateTransformations=[
                            v05.ScaleTransformation(scale=[1.0] * len(axes))
                        ],
                    )
                ],
            )
        ],
    )

    write_image(
        path,
        image,
        data,
        chunks=chunks,
        overwrite=True,
    )

    return path


def ngff_multi_position(
    path: str,
    n_positions: int = 3,
    shape: Mapping[str, int] = DEFAULT_SINGLE_SHAPE,
    dtype: str = "uint8",
) -> str:
    """Create a multi-position OME-NGFF (OME-Zarr) file.

    Creates a bioformats2raw-style layout with multiple positions/FOVs.
    Requires `yaozarrs[write-zarr]` to be installed.

    Parameters
    ----------
    path : str
        Path where the .ome.zarr directory will be created.
    n_positions : int, optional
        Number of positions/FOVs to create. Default is 3.
    shape : Mapping[str, int], optional
        Mapping of axis names to sizes. Must include "y" and "x".
        Optional axes: "t" (time), "c" (channel), "z" (z-slices).
        Default: {"t": 3, "c": 2, "z": 3, "y": 32, "x": 32}
    dtype : str, optional
        Data type for the array. Default is "uint8".

    Returns
    -------
    str
        Path to the created OME-Zarr store.

    Examples
    --------
    >>> import ndv
    >>> path = ndv.data.ngff_multi_position(
    ...     "/tmp/test_multipos.ome.zarr", n_positions=5
    ... )
    >>> ndv.imshow(path)  # Shows 'p' slider with 5 positions
    """
    import os

    try:
        import zarr
        from yaozarrs import v05
        from yaozarrs.write.v05 import write_image
    except ImportError as e:
        raise ImportError(
            "Please install yaozarrs with write support: "
            "pip install 'yaozarrs[write-zarr]'"
        ) from e

    shape_tuple = _shape_to_tuple(shape)
    axes = _build_axes(shape)
    chunks = tuple(
        1 if ax in ("t", "c") else min(shape[ax], 32)
        for ax in _AXIS_ORDER
        if ax in shape
    )

    rng = np.random.default_rng(42)

    # Create root zarr group with bioformats2raw metadata
    root = zarr.open_group(path, mode="w")
    root.attrs["bioformats2raw.layout"] = 3

    # Write each position
    for pos_idx in range(n_positions):
        data = rng.integers(0, np.iinfo(dtype).max // 4, size=shape_tuple, dtype=dtype)

        image = v05.Image(
            multiscales=[
                v05.Multiscale(
                    name=f"position_{pos_idx}",
                    axes=axes,
                    datasets=[
                        v05.Dataset(
                            path="0",
                            coordinateTransformations=[
                                v05.ScaleTransformation(scale=[1.0] * len(axes))
                            ],
                        )
                    ],
                )
            ],
        )

        pos_path = os.path.join(path, str(pos_idx))
        write_image(
            pos_path,
            image,
            data,
            chunks=chunks,
            overwrite=True,
        )

    return path


def ngff_plate(
    path: str,
    n_rows: int = 2,
    n_cols: int = 4,
    n_fovs: int = 1,
    shape: Mapping[str, int] = DEFAULT_SINGLE_SHAPE,
    dtype: str = "uint16",
) -> str:
    """Create an HCS plate OME-NGFF (OME-Zarr) file.

    Creates a high-content screening plate layout with wells and fields of view.
    Requires `yaozarrs[write]` to be installed.

    Parameters
    ----------
    path : str
        Path where the .ome.zarr directory will be created.
    n_rows : int, optional
        Number of well rows (A, B, C, ...). Default is 2.
    n_cols : int, optional
        Number of well columns (1, 2, 3, ...). Default is 4.
    n_fovs : int, optional
        Number of fields of view per well. Default is 1.
    shape : Mapping[str, int], optional
        Mapping of axis names to sizes. Must include "y" and "x".
        Optional axes: "t" (time), "c" (channel), "z" (z-slices).
        Default: {"t": 3, "c": 2, "z": 3, "y": 32, "x": 32}
    dtype : str, optional
        Data type for the array. Default is "uint16".

    Returns
    -------
    str
        Path to the created OME-Zarr store.

    Examples
    --------
    >>> import ndv
    >>> path = ndv.data.ngff_plate(
    ...     "/tmp/test_plate.ome.zarr", n_rows=2, n_cols=3, n_fovs=2
    ... )
    >>> ndv.imshow(path)  # Shows 'p' slider with 12 positions (2x3x2)
    """
    try:
        from yaozarrs import v05
        from yaozarrs.write.v05 import write_plate
    except ImportError as e:
        raise ImportError(
            "Please install yaozarrs with write support: pip install 'yaozarrs[write]'"
        ) from e

    shape_tuple = _shape_to_tuple(shape)
    axes = _build_axes(shape)
    chunks = tuple(
        1 if ax in ("t", "c") else min(shape[ax], 64)
        for ax in _AXIS_ORDER
        if ax in shape
    )

    rng = np.random.default_rng(42)

    # Build plate metadata
    row_names = [chr(65 + i) for i in range(n_rows)]  # A, B, C, ...
    col_names = [str(i + 1) for i in range(n_cols)]  # 1, 2, 3, ... (1-indexed)

    plate = v05.Plate(
        plate=v05.PlateDef(
            name="example_plate",
            rows=[v05.Row(name=r) for r in row_names],
            columns=[v05.Column(name=c) for c in col_names],
            wells=[
                v05.PlateWell(
                    path=f"{row}/{col}",
                    rowIndex=row_idx,
                    columnIndex=col_idx,
                )
                for row_idx, row in enumerate(row_names)
                for col_idx, col in enumerate(col_names)
            ],
            field_count=n_fovs,
        )
    )

    # Prepare images dictionary with (row, col, fov) keys
    images = {}
    for row in row_names:
        for col in col_names:
            for fov_idx in range(n_fovs):
                data = rng.integers(
                    0, np.iinfo(dtype).max // 4, size=shape_tuple, dtype=dtype
                )

                image = v05.Image(
                    multiscales=[
                        v05.Multiscale(
                            name=f"{row}/{col}_fov_{fov_idx}",
                            axes=axes,
                            datasets=[
                                v05.Dataset(
                                    path="0",
                                    coordinateTransformations=[
                                        v05.ScaleTransformation(scale=[1.0] * len(axes))
                                    ],
                                )
                            ],
                        )
                    ],
                )

                images[(row, col, str(fov_idx))] = (image, [data])

    write_plate(
        path,
        images,
        plate=plate,
        chunks=chunks,
        overwrite=True,
    )

    return path


def _build_axes(shape: Mapping[str, int]) -> list:
    """Build yaozarrs v05 axes from shape mapping in canonical order."""
    from yaozarrs import v05

    axes = []
    for axis in _AXIS_ORDER:
        if axis not in shape:
            continue
        if axis == "t":
            axes.append(v05.TimeAxis(name="t", type="time", unit="millisecond"))
        elif axis == "c":
            axes.append(v05.ChannelAxis(name="c", type="channel"))
        else:  # z, y, x are all space axes
            axes.append(v05.SpaceAxis(name=axis, type="space", unit="micrometer"))
    return axes


def _shape_to_tuple(shape: Mapping[str, int]) -> tuple[int, ...]:
    """Convert shape mapping to tuple in canonical axis order."""
    return tuple(shape[axis] for axis in _AXIS_ORDER if axis in shape)
