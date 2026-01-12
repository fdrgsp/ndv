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

    from yaozarrs import ZarrArray, ZarrGroup, v04, v05

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
        """Check if object is an OME-Zarr store or path."""
        # Check if it's a ZarrGroup with OME metadata
        try:
            from yaozarrs._zarr import ZarrGroup

            if isinstance(obj, ZarrGroup):
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

                return yaozarrs.open_group(obj).ome_metadata() is not None
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

        if self._is_multiposition:
            pos_path, _ = self._positions[0]
            array_path = self._first_array_path.replace(f"{pos_path}/", "")
            array_node = self._get_zarr_array(array_path, self._zarr_group[pos_path])
        else:
            array_node = self._get_zarr_array(self._first_array_path)

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

    def _get_zarr_array(self, path: str, parent: ZarrGroup | None = None) -> ZarrArray:
        """Get and validate a ZarrArray at the given path.

        Parameters
        ----------
        path : str
            Path to the array within the group.
        parent : ZarrGroup, optional
            Parent group to search in. Defaults to self._zarr_group.

        Returns
        -------
        ZarrArray
            The validated zarr array.

        Raises
        ------
        ValueError
            If path doesn't exist or isn't a ZarrArray.
        """
        from yaozarrs._zarr import ZarrArray as ZArray

        group = parent if parent is not None else self._zarr_group
        if path not in group:
            raise ValueError(f"Path {path} not found in group")
        node = group[path]
        if not isinstance(node, ZArray):
            raise ValueError(f"Expected ZarrArray at {path}")
        return node

    def _get_first_multiscale(self, metadata: Any) -> v04.Multiscale | v05.Multiscale:
        """Extract the first multiscale from metadata, validating its existence."""
        if not hasattr(metadata, "multiscales") or not metadata.multiscales:
            raise ValueError("No multiscales found in metadata")
        multiscale = metadata.multiscales[0]
        if not multiscale.datasets:
            raise ValueError("No datasets found in multiscale")
        return multiscale

    def _get_array_shape(self, array_node: ZarrArray, path: str) -> tuple[int, ...]:
        """Get array shape, raising if unavailable."""
        if (shape := array_node.metadata.shape) is None:
            raise ValueError(f"Array at {path} has no shape")
        return tuple(shape)

    def _extract_dims(
        self, multiscale: v04.Multiscale | v05.Multiscale, ndim: int
    ) -> tuple[Hashable, ...]:
        """Extract dimension names from multiscale axes or generate numbered dims."""
        if hasattr(multiscale, "axes") and multiscale.axes:
            return tuple(axis.name for axis in multiscale.axes)
        return tuple(range(ndim))

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

        multiscale = self._get_first_multiscale(self._metadata)
        dataset_path = multiscale.datasets[0].path
        self._first_array_path = dataset_path

        array_node = self._get_zarr_array(dataset_path)
        shape = self._get_array_shape(array_node, dataset_path)

        self._dims = self._extract_dims(multiscale, len(shape))
        self._coords = {dim: range(size) for dim, size in zip(self._dims, shape)}

    def _load_first_position_metadata(self) -> None:
        """Load metadata from first position to determine shape and axes."""
        if not self._positions:
            raise ValueError("No positions to load metadata from")

        first_pos_path, res_idx = self._positions[0]
        pos_group = self._zarr_group[first_pos_path]
        pos_metadata = pos_group.ome_metadata()

        multiscale = self._get_first_multiscale(pos_metadata)
        if res_idx >= len(multiscale.datasets):
            raise ValueError(
                f"Resolution index {res_idx} out of range for position {first_pos_path}"
            )

        dataset_path = multiscale.datasets[res_idx].path
        full_array_path = f"{first_pos_path}/{dataset_path}"
        self._first_array_path = full_array_path

        array_node = self._get_zarr_array(dataset_path, pos_group)
        shape = self._get_array_shape(array_node, full_array_path)

        inner_dims = self._extract_dims(multiscale, len(shape))
        self._dims = ("p", *inner_dims)
        self._coords = {
            "p": range(len(self._positions)),
            **{dim: range(size) for dim, size in zip(inner_dims, shape)},
        }

    def _isel_single(self, indexers: Mapping[int, int | slice]) -> np.ndarray:
        """Select data for single-position images."""
        if self._first_array_path is None:
            raise ValueError("No array path available")

        array_node = self._get_zarr_array(self._first_array_path)
        idx_tuple = tuple(indexers.get(i, slice(None)) for i in range(len(self._dims)))
        return self._read_array_data(array_node, idx_tuple)

    def _isel_multiposition(self, indexers: Mapping[int, int | slice]) -> np.ndarray:
        """Select data for multi-position images."""
        pos_idx, keep_pos_dim = self._resolve_position_index(indexers.get(0))
        pos_path, res_idx = self._positions[pos_idx]

        pos_group = self._zarr_group[pos_path]
        multiscale = self._get_first_multiscale(pos_group.ome_metadata())
        dataset_path = multiscale.datasets[res_idx].path
        array_node = self._get_zarr_array(dataset_path, pos_group)

        # Build index tuple for inner dimensions (skip position dimension)
        inner_indexers = {
            i - 1: indexers[i] for i in range(1, len(self._dims)) if i in indexers
        }
        idx_tuple = tuple(
            inner_indexers.get(i, slice(None)) for i in range(len(self._dims) - 1)
        )

        data = self._read_array_data(array_node, idx_tuple)
        return data[np.newaxis, ...] if keep_pos_dim else data

    def _resolve_position_index(self, pos_idx: int | slice | None) -> tuple[int, bool]:
        """Resolve position index and determine if dimension should be kept.

        Returns
        -------
        tuple[int, bool]
            (resolved_index, keep_dimension_flag)
        """
        keep_pos_dim = False
        if pos_idx is None:
            pos_idx = 0
        elif isinstance(pos_idx, slice):
            pos_idx = pos_idx.start if pos_idx.start is not None else 0
            keep_pos_dim = True

        if not isinstance(pos_idx, int):
            raise ValueError("Position index must be an integer")
        if not 0 <= pos_idx < len(self._positions):
            raise IndexError(
                f"Position index {pos_idx} out of range [0, {len(self._positions)})"
            )
        return pos_idx, keep_pos_dim

    def _read_array_data(self, array_node: ZarrArray, idx_tuple: tuple) -> np.ndarray:
        """Read array data using tensorstore or zarr-python fallback."""
        try:
            return np.asarray(array_node.to_tensorstore()[idx_tuple].read().result())
        except ImportError:
            return np.asarray(array_node.to_zarr_python()[idx_tuple])
