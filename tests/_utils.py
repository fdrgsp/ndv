from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

_AXIS_ORDER = ("t", "c", "z", "y", "x")
DEFAULT_SHAPE: dict[str, int] = {"t": 3, "c": 2, "z": 3, "y": 32, "x": 32}


def _shape_to_tuple(shape: Mapping[str, int]) -> tuple[int, ...]:
    return tuple(shape[axis] for axis in _AXIS_ORDER if axis in shape)


def _build_axes(shape: Mapping[str, int]) -> list:
    from yaozarrs import v05

    axes = []
    for axis in _AXIS_ORDER:
        if axis not in shape:
            continue
        if axis == "t":
            axes.append(v05.TimeAxis(name="t", type="time", unit="millisecond"))
        elif axis == "c":
            axes.append(v05.ChannelAxis(name="c", type="channel"))
        else:
            axes.append(v05.SpaceAxis(name=axis, type="space", unit="micrometer"))
    return axes


def _chunks(shape: Mapping[str, int], max_chunk: int = 64) -> tuple[int, ...]:
    return tuple(
        1 if ax in ("t", "c") else min(shape[ax], max_chunk)
        for ax in _AXIS_ORDER
        if ax in shape
    )


def _make_dataset(axes: list) -> list:
    from yaozarrs import v05

    scale = [1.0] * len(axes)
    return [
        v05.Dataset(
            path="0", coordinateTransformations=[v05.ScaleTransformation(scale=scale)]
        )
    ]


def ngff_single_position(
    path: str,
    shape: Mapping[str, int] = DEFAULT_SHAPE,
    dtype: str = "uint8",
) -> str:
    """Create a single-position OME-Zarr file for testing.

    Parameters
    ----------
    path : str
        Path where the .ome.zarr directory will be created.
    shape : Mapping[str, int], optional
        Mapping of axis names to sizes. Must include "y" and "x".
        Optional axes: "t" (time), "c" (channel), "z" (z-slices).
        Default: {"t": 3, "c": 2, "z": 3, "y": 32, "x": 32}
    dtype : str, optional
        Data type for the array. Default is "uint8".

    """
    from yaozarrs import v05
    from yaozarrs.write.v05 import write_image

    shape_tuple = _shape_to_tuple(shape)
    axes = _build_axes(shape)
    rng = np.random.default_rng(42)
    data = rng.integers(0, np.iinfo(dtype).max // 4, size=shape_tuple, dtype=dtype)
    image = v05.Image(
        multiscales=[
            v05.Multiscale(name="single", axes=axes, datasets=_make_dataset(axes))
        ]
    )
    write_image(path, image, data, chunks=_chunks(shape), overwrite=True)
    return path


def ngff_multi_position(
    path: str,
    n_positions: int = 3,
    shape: Mapping[str, int] = DEFAULT_SHAPE,
    dtype: str = "uint8",
) -> str:
    """Create a multi-position OME-Zarr file for testing.

    Parameters
    ----------
    path : str
        Path where the .ome.zarr directory will be created.
    n_positions : int, optional
        Number of positions to create. Default is 3.
    shape : Mapping[str, int], optional
        Mapping of axis names to sizes. Must include "y" and "x".
        Optional axes: "t" (time), "c" (channel), "z" (z-slices).
        Default: {"t": 3, "c": 2, "z": 3, "y": 32, "x": 32}
    dtype : str, optional
        Data type for the array. Default is "uint8".
    """
    import os

    import zarr
    from yaozarrs import v05
    from yaozarrs.write.v05 import write_image

    shape_tuple = _shape_to_tuple(shape)
    axes = _build_axes(shape)
    chunks = _chunks(shape, 32)
    rng = np.random.default_rng(42)
    root = zarr.open_group(path, mode="w")
    root.attrs["bioformats2raw.layout"] = 3

    for pos_idx in range(n_positions):
        data = rng.integers(0, np.iinfo(dtype).max // 4, size=shape_tuple, dtype=dtype)
        image = v05.Image(
            multiscales=[
                v05.Multiscale(
                    name=f"pos_{pos_idx}", axes=axes, datasets=_make_dataset(axes)
                )
            ]
        )
        pos_path = os.path.join(path, str(pos_idx))
        write_image(pos_path, image, data, chunks=chunks, overwrite=True)
    return path


def ngff_plate(
    path: str,
    n_rows: int = 2,
    n_cols: int = 4,
    n_fovs: int = 1,
    shape: Mapping[str, int] = DEFAULT_SHAPE,
    dtype: str = "uint8",
) -> str:
    """Create an HCS plate OME-Zarr file for testing.

    Parameters
    ----------
    path : str
        Path where the .ome.zarr directory will be created.
    n_rows : int, optional
        Number of plate rows. Default is 2.
    n_cols : int, optional
        Number of plate columns. Default is 4.
    n_fovs : int, optional
        Number of fields of view per well. Default is 1.
    shape : Mapping[str, int], optional
        Mapping of axis names to sizes. Must include "y" and "x".
        Optional axes: "t" (time), "c" (channel), "z" (z-slices).
        Default: {"t": 3, "c": 2, "z": 3, "y": 32, "x": 32}
    dtype : str, optional
        Data type for the array. Default is "uint8".
    """
    from yaozarrs import v05
    from yaozarrs.write.v05 import write_plate

    shape_tuple = _shape_to_tuple(shape)
    axes = _build_axes(shape)
    chunks = _chunks(shape)
    rng = np.random.default_rng(42)
    row_names = [chr(65 + i) for i in range(n_rows)]
    col_names = [str(i + 1) for i in range(n_cols)]

    plate = v05.Plate(
        plate=v05.PlateDef(
            name="plate",
            rows=[v05.Row(name=r) for r in row_names],
            columns=[v05.Column(name=c) for c in col_names],
            wells=[
                v05.PlateWell(
                    path=f"{row}/{col}", rowIndex=row_idx, columnIndex=col_idx
                )
                for row_idx, row in enumerate(row_names)
                for col_idx, col in enumerate(col_names)
            ],
            field_count=n_fovs,
        )
    )

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
                            name=f"{row}{col}_{fov_idx}",
                            axes=axes,
                            datasets=_make_dataset(axes),
                        )
                    ]
                )
                images[(row, col, str(fov_idx))] = (image, [data])

    write_plate(path, images, plate=plate, chunks=chunks, overwrite=True)
    return path
