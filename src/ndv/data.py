"""Sample data for testing and examples."""

# pyright: reportMissingImports=none
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "astronaut",
    "cat",
    "cells3d",
    "cosem_dataset",
    "nd_sine_wave",
    "ngff_multi_position",
    "ngff_plate",
    "ngff_single_position",
]


def nd_sine_wave(
    shape: tuple[int, int, int, int, int] = (10, 3, 5, 512, 512),
    amplitude: float = 240,
    base_frequency: float = 5,
) -> np.ndarray:
    """5D dataset: `(10, 3, 5, 512, 512)`, float64."""
    # Unpack the dimensions
    if not len(shape) == 5:
        raise ValueError("Shape must have 5 dimensions")
    angle_dim, freq_dim, phase_dim, ny, nx = shape

    # Create an empty array to hold the data
    output = np.zeros(shape)

    # Define spatial coordinates for the last two dimensions
    half_per = base_frequency * np.pi
    x = np.linspace(-half_per, half_per, nx)
    y = np.linspace(-half_per, half_per, ny)
    y, x = np.meshgrid(y, x)

    # Iterate through each parameter in the higher dimensions
    for phase_idx in range(phase_dim):
        for freq_idx in range(freq_dim):
            for angle_idx in range(angle_dim):
                # Calculate phase and frequency
                phase = np.pi / phase_dim * phase_idx
                frequency = 1 + (freq_idx * 0.1)  # Increasing frequency with each step

                # Calculate angle
                angle = np.pi / angle_dim * angle_idx
                # Rotate x and y coordinates
                xr = np.cos(angle) * x - np.sin(angle) * y

                # Compute the sine wave
                sine_wave = (amplitude * 0.5) * np.sin(frequency * xr + phase)
                sine_wave += amplitude * 0.5

                # Assign to the output array
                output[angle_idx, freq_idx, phase_idx] = sine_wave

    return output.astype(np.float32)


def cells3d() -> np.ndarray:
    """Load cells3d from scikit-image `(60, 2, 256, 256)` uint16.

    Requires `imageio and tifffile` to be installed.
    """
    try:
        from imageio.v2 import volread
    except ImportError as e:
        raise ImportError(
            "Please `pip install imageio[tifffile]` to load cells3d"
        ) from e

    url = "https://gitlab.com/scikit-image/data/-/raw/2cdc5ce89b334d28f06a58c9f0ca21aa6992a5ba/cells3d.tif"
    data = np.asarray(volread(url))

    # this data has been stretched to 16 bit, and lacks certain intensity values
    # add a small random integer to each pixel ... so the histogram is not silly
    data = (data + np.random.randint(-24, 24, data.shape)).clip(0, 65535)
    return data.astype(np.uint16)


def cat() -> np.ndarray:
    """Load RGB cat data `(300, 451, 3)`, uint8.

    Requires [imageio](https://pypi.org/project/imageio/) to be installed.
    """
    return _imread("imageio:chelsea.png")


def astronaut() -> np.ndarray:
    """Load RGB data `(512, 512, 3)`, uint8.

    Requires [imageio](https://pypi.org/project/imageio/) to be installed.
    """
    return _imread("imageio:astronaut.png")


def _imread(uri: str) -> np.ndarray:
    try:
        import imageio.v3 as iio
    except ImportError:
        raise ImportError("Please install imageio fetch data") from None
    return iio.imread(uri)  # type: ignore [no-any-return]


def cosem_dataset(
    uri: str = "",
    dataset: str = "jrc_hela-3",
    label: str = "er-mem_pred",
    level: int = 4,
) -> Any:
    """Load a dataset from the COSEM/OpenOrganelle project.

    Search for available options at: <https://openorganelle.janelia.org/datasets>

    Requires [tensorstore](https://pypi.org/project/tensorstore/) to be installed.

    Parameters
    ----------
    uri : str, optional
        The URI of the dataset to load. If not provided, the default URI is
        `f"{dataset}/{dataset}.n5/labels/{label}/s{level}/"`.
    dataset : str, optional
        The name of the dataset to load. Default is "jrc_hela-3".
    label : str, optional
        The label to load. Default is "er-mem_pred".
    level : int, optional
        The pyramid level to load. Default is 4.
    """
    try:
        import tensorstore as ts
    except ImportError:
        raise ImportError("Please install tensorstore to fetch cosem data") from None

    if not uri:
        uri = f"{dataset}/{dataset}.n5/labels/{label}/s{level}/"

    ts_array = ts.open(
        {
            "driver": "n5",
            "kvstore": {
                "driver": "s3",
                "bucket": "janelia-cosem-datasets",
                "path": uri,
            },
            # 1GB cache... but i don't think it's working
            "cache_pool": {"total_bytes_limit": 1e9},
        },
    ).result()
    ts_array = ts_array[ts.d[:].label["z", "y", "x"]]
    return ts_array[ts.d[("y", "x", "z")].transpose[:]]


def rgba() -> np.ndarray:
    """3D RGBA dataset: `(256, 256, 256, 4)`, uint8."""
    img = np.zeros((256, 256, 256, 4), dtype=np.uint8)

    # R,G,B are simple
    for i in range(256):
        img[:, i, :, 0] = i  # Red
        img[:, i, :, 2] = 255 - i  # Blue
    for j in range(256):
        img[:, :, j, 1] = j  # Green

    # Alpha is a bit trickier - requires a meshgrid for efficient computation
    x, y, z = np.meshgrid(np.arange(256), np.arange(256), np.arange(256), indexing="ij")
    alpha = np.sqrt((x - 128) ** 2 + (y - 128) ** 2 + (z - 128) ** 2)
    img[:, :, :, 3] = np.clip(alpha, 0, 255)

    return img


def ngff_single_position(
    path: str,
    shape: tuple[int, int, int, int, int] = (3, 2, 4, 64, 64),
    dtype: str = "uint16",
) -> str:
    """Create a single-position OME-NGFF (OME-Zarr) file.

    Creates a standard multiscale OME-Zarr image with axes (t, c, z, y, x).
    Requires `yaozarrs[write]` to be installed.

    Parameters
    ----------
    path : str
        Path where the .ome.zarr directory will be created.
    shape : tuple[int, int, int, int, int], optional
        Shape of the data as (T, C, Z, Y, X). Default is (3, 2, 4, 64, 64).
    dtype : str, optional
        Data type for the array. Default is "uint16".

    Returns
    -------
    str
        Path to the created OME-Zarr store.

    Examples
    --------
    >>> import ndv
    >>> path = ndv.data.ngff_single_position("/tmp/test_single.ome.zarr")
    >>> ndv.imshow(path)
    """
    try:
        from yaozarrs import v05
        from yaozarrs.write.v05 import write_image
    except ImportError as e:
        raise ImportError(
            "Please install yaozarrs with write support: pip install 'yaozarrs[write]'"
        ) from e

    # Create random data
    rng = np.random.default_rng(42)
    data = rng.integers(0, np.iinfo(dtype).max // 4, size=shape, dtype=dtype)

    # Create OME-Zarr Image metadata
    image = v05.Image(
        multiscales=[
            v05.Multiscale(
                name="single_position_example",
                axes=[
                    v05.TimeAxis(name="t", type="time", unit="millisecond"),
                    v05.ChannelAxis(name="c", type="channel"),
                    v05.SpaceAxis(name="z", type="space", unit="micrometer"),
                    v05.SpaceAxis(name="y", type="space", unit="micrometer"),
                    v05.SpaceAxis(name="x", type="space", unit="micrometer"),
                ],
                datasets=[
                    v05.Dataset(
                        path="0",
                        coordinateTransformations=[
                            v05.ScaleTransformation(scale=[1.0, 1.0, 1.0, 1.0, 1.0])
                        ],
                    )
                ],
            )
        ],
    )

    # Write the image with data
    write_image(
        path,
        image,
        data,
        chunks=(1, 1, 1, 64, 64),
        compression="blosc-zstd",
        overwrite=True,
    )

    return path


def ngff_multi_position(
    path: str,
    n_positions: int = 3,
    shape: tuple[int, int, int, int] = (2, 2, 32, 32),
    dtype: str = "uint8",
) -> str:
    """Create a multi-position OME-NGFF (OME-Zarr) file.

    Creates a bioformats2raw-style layout with multiple positions/FOVs.
    Each position has axes (t, c, y, x). Requires `yaozarrs[write-zarr]`
    to be installed.

    Parameters
    ----------
    path : str
        Path where the .ome.zarr directory will be created.
    n_positions : int, optional
        Number of positions/FOVs to create. Default is 3.
    shape : tuple[int, int, int, int], optional
        Shape of each position as (T, C, Y, X). Default is (2, 2, 32, 32).
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
    try:
        import zarr
        from yaozarrs import v05
        from yaozarrs.write.v05 import write_image
    except ImportError as e:
        raise ImportError(
            "Please install yaozarrs with write support: "
            "pip install 'yaozarrs[write-zarr]'"
        ) from e
    import os

    rng = np.random.default_rng(42)

    # Create root zarr group with bioformats2raw metadata
    root = zarr.open_group(path, mode="w")
    root.attrs["bioformats2raw.layout"] = 3

    # Write each position using write_image
    for pos_idx in range(n_positions):
        # Create random data for this position
        data = rng.integers(0, np.iinfo(dtype).max // 4, size=shape, dtype=dtype)

        # Create Image metadata for this position
        image = v05.Image(
            multiscales=[
                v05.Multiscale(
                    name=f"position_{pos_idx}",
                    axes=[
                        v05.TimeAxis(name="t", type="time", unit="millisecond"),
                        v05.ChannelAxis(name="c", type="channel"),
                        v05.SpaceAxis(name="y", type="space", unit="micrometer"),
                        v05.SpaceAxis(name="x", type="space", unit="micrometer"),
                    ],
                    datasets=[
                        v05.Dataset(
                            path="0",
                            coordinateTransformations=[
                                v05.ScaleTransformation(scale=[1.0, 1.0, 1.0, 1.0])
                            ],
                        )
                    ],
                )
            ],
        )

        # Write this position as a subgroup
        pos_path = os.path.join(path, str(pos_idx))
        write_image(
            pos_path,
            image,
            data,
            chunks=(1, 1, 32, 32),
            compression="blosc-zstd",
            overwrite=True,
        )

    return path


def ngff_plate(
    path: str,
    n_rows: int = 2,
    n_cols: int = 4,
    n_fovs: int = 1,
    shape: tuple[int, int, int, int] = (2, 3, 64, 64),
    dtype: str = "uint16",
) -> str:
    """Create an HCS plate OME-NGFF (OME-Zarr) file.

    Creates a high-content screening plate layout with wells and fields of view.
    Each FOV has axes (c, z, y, x). Requires `yaozarrs[write]` to be installed.

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
    shape : tuple[int, int, int, int], optional
        Shape of each FOV as (C, Z, Y, X). Default is (2, 3, 64, 64).
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
                # Create random data for this FOV
                data = rng.integers(
                    0, np.iinfo(dtype).max // 4, size=shape, dtype=dtype
                )

                image = v05.Image(
                    multiscales=[
                        v05.Multiscale(
                            name=f"{row}/{col}_fov_{fov_idx}",
                            axes=[
                                v05.ChannelAxis(name="c", type="channel"),
                                v05.SpaceAxis(
                                    name="z", type="space", unit="micrometer"
                                ),
                                v05.SpaceAxis(
                                    name="y", type="space", unit="micrometer"
                                ),
                                v05.SpaceAxis(
                                    name="x", type="space", unit="micrometer"
                                ),
                            ],
                            datasets=[
                                v05.Dataset(
                                    path="0",
                                    coordinateTransformations=[
                                        v05.ScaleTransformation(
                                            scale=[1.0, 1.0, 1.0, 1.0]
                                        )
                                    ],
                                )
                            ],
                        )
                    ],
                )

                # Add to images dict with (row, col, fov) key
                images[(row, col, str(fov_idx))] = (image, [data])

    # Write plate
    write_plate(
        path,
        images,
        plate=plate,
        chunks=(1, 1, 64, 64),
        compression="blosc-zstd",
        overwrite=True,
    )

    return path
