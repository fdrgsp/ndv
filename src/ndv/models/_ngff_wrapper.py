"""DataWrapper for OME-NGFF (OME-Zarr) stores with multi-position and plate support.

This module provides support for opening OME-Zarr stores that follow the
NGFF specification (version 0.4 and 0.5), including:
- Single-position images
- Multi-position (multi-FOV) images
- High-content screening plates with wells and FOVs

Requires the yaozarrs[io] package to be installed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ndv.models._data_wrapper import DataWrapper

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence
    from typing import TypeGuard

    from yaozarrs import ZarrGroup, v04, v05

logger = logging.getLogger(__name__)


class NGFFWrapper(DataWrapper["ZarrGroup"]):
    """Wrapper for OME-NGFF/OME-Zarr stores with multi-position/plate support.

    This wrapper handles:
    - Single-position images (standard multiscale)
    - Multi-position/multi-FOV datasets (bioformats2raw layout)
    - HCS plates with wells and fields of view

    For multi-position datasets, a "p" dimension is added as the first
    dimension, allowing navigation through positions using the ndv position slider.

    For plates, positions are ordered by: well (row-major), then FOV within well.

    Examples
    --------
    ```python
    import ndv

    # Works with paths - automatically detected as NGFF
    ndv.imshow("/path/to/data.ome.zarr")

    # Or with yaozarrs ZarrGroup objects
    import yaozarrs

    group = yaozarrs.open_group("/path/to/data.ome.zarr")
    ndv.imshow(group)
    ```
    """

    PRIORITY = 45  # Check before generic zarr arrays but after more specific types

    def __init__(self, data: ZarrGroup | str) -> None:
        """Initialize NGFF wrapper.

        Parameters
        ----------
        data : ZarrGroup | str
            yaozarrs ZarrGroup or string path to an OME-Zarr store.
        """
        import yaozarrs

        # Convert string paths to ZarrGroup
        self._zarr_group = yaozarrs.open_group(data) if isinstance(data, str) else data
        self._metadata = self._zarr_group.ome_metadata()
        self._positions: list[tuple[str, int]] = []  # (path, resolution_index)
        self._dims: tuple[Hashable, ...] = ()
        self._coords: dict[Hashable, Sequence] = {}
        self._is_multiposition = False
        self._first_array_path: str | None = None

        # Detect structure and build position list
        self._detect_structure()

        # Initialize parent with the group
        super().__init__(self._zarr_group)

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[ZarrGroup]:
        """Check if object is an OME-Zarr store or path.

        Parameters
        ----------
        obj : Any
            Object to check.

        Returns
        -------
        bool
            True if obj is a yaozarrs.ZarrGroup or path to an OME-Zarr store.
        """
        # Check if it's a ZarrGroup with OME metadata
        try:
            from yaozarrs._zarr import ZarrGroup

            if isinstance(obj, ZarrGroup):
                # Check if it has OME metadata
                try:
                    return obj.ome_metadata() is not None
                except Exception:
                    return False
        except ImportError:
            pass

        # Check if it's a string path ending in .ome.zarr
        if isinstance(obj, str):
            if obj.endswith(".ome.zarr"):
                return True
            # Try to open and check for OME metadata
            try:
                import yaozarrs

                group = yaozarrs.open_group(obj)
                return group.ome_metadata() is not None
            except Exception:
                return False

        return False

    @property
    def dims(self) -> tuple[Hashable, ...]:
        """Dimension labels for the data."""
        return self._dims

    @property
    def coords(self) -> Mapping[Hashable, Sequence]:
        """Coordinates for the data."""
        return self._coords

    @property
    def dtype(self) -> np.dtype:
        """Dtype of the data."""
        if self._first_array_path is None:
            raise ValueError("No array path available")

        if not self._is_multiposition:
            array_node = self._zarr_group[self._first_array_path]
        else:
            # Navigate through position
            pos_path, _ = self._positions[0]
            pos_group = self._zarr_group[pos_path]
            array_path = self._first_array_path.replace(f"{pos_path}/", "")
            array_node = pos_group[array_path]

        from yaozarrs._zarr import ZarrArray

        if not isinstance(array_node, ZarrArray):
            raise ValueError("Expected ZarrArray")

        return np.dtype(array_node.dtype)  # type: ignore[no-any-return]

    def isel(self, indexers: Mapping[int, int | slice]) -> np.ndarray:
        """Select data by integer indices.

        Parameters
        ----------
        indexers : Mapping[int, int | slice]
            Mapping from dimension index to slice/index.

        Returns
        -------
        np.ndarray
            Selected data as numpy array.
        """
        if not self._is_multiposition:
            return self._isel_single(indexers)
        return self._isel_multiposition(indexers)

    def _detect_structure(self) -> None:
        """Detect NGFF structure and dispatch to appropriate parser."""
        from yaozarrs import v04, v05

        metadata = self._metadata

        # Check structure type in priority order
        if isinstance(metadata, (v04.Plate, v05.Plate)):
            self._parse_plate()
        elif isinstance(metadata, (v04.Well, v05.Well)):
            self._parse_well()
        elif hasattr(metadata, "multiscales") and metadata.multiscales:
            if self._has_bioformats2raw_layout():
                self._parse_bioformats2raw()
            else:
                self._parse_single_image()
        elif (
            hasattr(metadata, "bioformats2raw_layout")
            and metadata.bioformats2raw_layout
        ):
            self._parse_bioformats2raw()
        else:
            raise ValueError(
                f"Could not determine NGFF structure for {self._zarr_group.store_path}"
            )

    def _has_bioformats2raw_layout(self) -> bool:
        """Check if zarr group has bioformats2raw layout."""
        attrs = self._zarr_group.attrs
        return "bioformats2raw.layout" in attrs or (
            "ome" in attrs and "bioformats2raw.layout" in attrs["ome"]
        )

    def _parse_plate(self) -> None:
        """Parse plate structure and collect all FOV positions."""
        logger.debug("Parsing plate structure")
        self._is_multiposition = True

        metadata = cast("v04.Plate | v05.Plate", self._metadata)
        plate_def = metadata.plate

        # Collect FOVs from all wells
        for well_ref in plate_def.wells:
            well_path = well_ref.path
            if well_path not in self._zarr_group:
                logger.warning(f"Well {well_path} not found in plate")
                continue

            well_group = self._zarr_group[well_path]
            well_metadata = well_group.ome_metadata()

            if not hasattr(well_metadata, "well"):
                logger.warning(f"Well {well_path} has no well metadata")
                continue

            # Add FOVs from this well (use first resolution)
            self._positions.extend(
                (f"{well_path}/{fov.path}", 0) for fov in well_metadata.well.images
            )

        if not self._positions:
            raise ValueError("No FOV positions found in plate")

        self._load_first_position_metadata()

    def _parse_well(self) -> None:
        """Parse well structure and collect all FOV positions."""
        logger.debug("Parsing well structure")
        self._is_multiposition = True

        metadata = cast("v04.Well | v05.Well", self._metadata)
        # Collect all FOVs in this well (use first resolution)
        self._positions = [(fov.path, 0) for fov in metadata.well.images]

        if not self._positions:
            raise ValueError("No FOV positions found in well")

        self._load_first_position_metadata()

    def _parse_bioformats2raw(self) -> None:
        """Parse bioformats2raw multi-position layout."""
        logger.debug("Parsing bioformats2raw multi-position layout")
        self._is_multiposition = True

        # Check for OME/series metadata
        if "OME" in self._zarr_group:
            ome_meta = self._zarr_group["OME"].metadata
            if "series" in ome_meta.attributes:
                self._positions = [(path, 0) for path in ome_meta.attributes["series"]]
            else:
                self._find_numbered_positions()
        else:
            self._find_numbered_positions()

        if not self._positions:
            raise ValueError("No positions found in bioformats2raw layout")

        self._load_first_position_metadata()

    def _find_numbered_positions(self) -> None:
        """Find numbered position subgroups (0, 1, 2, ...)."""
        i = 0
        while (path := str(i)) in self._zarr_group:
            child = self._zarr_group[path]
            if hasattr(child, "ome_metadata"):
                child_meta = child.ome_metadata()
                if hasattr(child_meta, "multiscales") and child_meta.multiscales:
                    self._positions.append((path, 0))
            i += 1

    def _parse_single_image(self) -> None:
        """Parse single multiscale image."""
        logger.debug("Parsing single-position image")
        self._is_multiposition = False

        # Use first multiscale, first resolution
        if not hasattr(self._metadata, "multiscales") or not self._metadata.multiscales:
            raise ValueError("No multiscales found in image")

        multiscale = self._metadata.multiscales[0]
        if not multiscale.datasets:
            raise ValueError("No datasets found in multiscale")

        dataset_path = multiscale.datasets[0].path
        self._first_array_path = dataset_path

        # Get array and validate
        if dataset_path not in self._zarr_group:
            raise ValueError(f"Dataset path {dataset_path} not found")

        from yaozarrs._zarr import ZarrArray

        array_node = self._zarr_group[dataset_path]
        if not isinstance(array_node, ZarrArray):
            raise ValueError(f"Expected ZarrArray at {dataset_path}")

        # Build dims from axes or fallback to numbered
        self._dims = (
            tuple(axis.name for axis in multiscale.axes)
            if hasattr(multiscale, "axes")
            else tuple(range(len(array_node.metadata.shape)))
        )

        # Build coords from shape
        if (shape := array_node.metadata.shape) is None:
            raise ValueError(f"Array at {dataset_path} has no shape")

        self._coords = {dim: range(size) for dim, size in zip(self._dims, shape)}

    def _load_first_position_metadata(self) -> None:
        """Load metadata from first position to determine shape and axes."""
        if not self._positions:
            raise ValueError("No positions to load metadata from")

        first_pos_path, res_idx = self._positions[0]

        # Navigate to position and get metadata
        pos_group = self._zarr_group[first_pos_path]
        pos_metadata = pos_group.ome_metadata()

        if not hasattr(pos_metadata, "multiscales") or not pos_metadata.multiscales:
            raise ValueError(f"Position {first_pos_path} has no multiscales")

        multiscale = pos_metadata.multiscales[0]
        if res_idx >= len(multiscale.datasets):
            raise ValueError(
                f"Resolution index {res_idx} out of range for position {first_pos_path}"
            )

        dataset_path = multiscale.datasets[res_idx].path
        full_array_path = f"{first_pos_path}/{dataset_path}"
        self._first_array_path = full_array_path

        # Get array and validate
        from yaozarrs._zarr import ZarrArray

        array_node = pos_group[dataset_path]
        if not isinstance(array_node, ZarrArray):
            raise ValueError(f"Expected ZarrArray at {full_array_path}")

        if (shape := array_node.metadata.shape) is None:
            raise ValueError(f"Array at {full_array_path} has no shape")

        # Build dims with "p" as first dimension
        inner_dims = (
            tuple(axis.name for axis in multiscale.axes)
            if hasattr(multiscale, "axes")
            else tuple(range(len(shape)))
        )
        self._dims = ("p", *inner_dims)

        # Build coords
        self._coords = {
            "p": range(len(self._positions)),
            **{dim: range(size) for dim, size in zip(inner_dims, shape)},
        }

    def _isel_single(self, indexers: Mapping[int, int | slice]) -> np.ndarray:
        """Select data for single-position images."""
        if self._first_array_path is None:
            raise ValueError("No array path available")

        from yaozarrs._zarr import ZarrArray

        array_node = self._zarr_group[self._first_array_path]
        if not isinstance(array_node, ZarrArray):
            raise ValueError(f"Expected ZarrArray at {self._first_array_path}")

        idx_tuple = tuple(indexers.get(i, slice(None)) for i in range(len(self._dims)))
        return self._read_array_data(array_node, idx_tuple)

    def _isel_multiposition(self, indexers: Mapping[int, int | slice]) -> np.ndarray:
        """Select data for multi-position images."""
        pos_idx = indexers.get(0)  # position is always dimension 0
        keep_pos_dim = False

        # Handle position index
        if pos_idx is None:
            pos_idx = 0
        elif isinstance(pos_idx, slice):
            pos_idx = pos_idx.start if pos_idx.start is not None else 0
            keep_pos_dim = True  # slice means preserve singleton dimension

        if not isinstance(pos_idx, int):
            raise ValueError("Position index must be an integer")

        if not 0 <= pos_idx < len(self._positions):
            raise IndexError(
                f"Position index {pos_idx} out of range [0, {len(self._positions)})"
            )

        pos_path, res_idx = self._positions[pos_idx]

        # Navigate to position and get dataset
        pos_group = self._zarr_group[pos_path]
        pos_metadata = pos_group.ome_metadata()

        if not hasattr(pos_metadata, "multiscales") or not pos_metadata.multiscales:
            raise ValueError(f"Position {pos_path} has no multiscales")

        dataset_path = pos_metadata.multiscales[0].datasets[res_idx].path

        from yaozarrs._zarr import ZarrArray

        array_node = pos_group[dataset_path]
        if not isinstance(array_node, ZarrArray):
            raise ValueError(f"Expected ZarrArray at {pos_path}/{dataset_path}")

        # Build index tuple for inner dimensions (skip position dimension)
        inner_indexers = {
            i - 1: indexers[i] for i in range(1, len(self._dims)) if i in indexers
        }
        idx_tuple = tuple(
            inner_indexers.get(i, slice(None)) for i in range(len(self._dims) - 1)
        )

        data = self._read_array_data(array_node, idx_tuple)
        return data[np.newaxis, ...] if keep_pos_dim else data

    def _read_array_data(self, array_node: Any, idx_tuple: tuple) -> np.ndarray:
        """Read array data using tensorstore or zarr-python."""
        try:
            # Try tensorstore first (better performance)
            return np.asarray(array_node.to_tensorstore()[idx_tuple].read().result())
        except ImportError:
            # Fall back to zarr-python (v3)
            return np.asarray(array_node.to_zarr_python()[idx_tuple])

