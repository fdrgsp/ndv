"""In this module, we provide built-in support for many array types."""

# pyright: reportMissingImports=none
from __future__ import annotations

import json
import logging
import sys
import warnings
from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, TypeVar

import numpy as np
import numpy.typing as npt
from psygnal import Signal

from ._ring_buffer import RingBuffer

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    from typing_extensions import TypeGuard
if TYPE_CHECKING:
    from collections.abc import Container, Iterator
    from typing import Union

    import dask.array.core as da
    import numpy.typing as npt
    import pydantic_core
    import pyopencl.array as cl_array
    import sparse
    import tensorstore as ts
    import torch
    import xarray as xr
    from pydantic import GetCoreSchemaHandler
    from typing_extensions import TypeAlias, TypeGuard

    Index: TypeAlias = Union[int, slice]


class SupportsIndexing(Protocol):
    def __getitem__(self, key: Index | tuple[Index, ...]) -> npt.ArrayLike: ...
    @property
    def shape(self) -> tuple[int, ...]: ...


ArrayT = TypeVar("ArrayT")
NPArrayLike = TypeVar("NPArrayLike", bound=SupportsIndexing)
_T = TypeVar("_T", bound=type)


def _recurse_subclasses(cls: _T) -> Iterator[_T]:
    for subclass in cls.__subclasses__():
        yield subclass
        yield from _recurse_subclasses(subclass)


class DataWrapper(Generic[ArrayT], ABC):
    """Interface for wrapping different array-like data types.

    [`DataWrapper.create()`][ndv.DataWrapper.create] is a factory method that returns a
    `DataWrapper` instance for the given data type. If your datastore type is not
    supported, you may implement a new `DataWrapper` subclass to handle your data type.
    To do this, import and subclass `DataWrapper`, and (minimally) implement the
    supports and isel methods. Ensure that your class is imported before the
    `DataWrapper.create` method is called, and it will be automatically detected and
    used to wrap your data.

    This base class provides basic support for numpy-like array types.  If the data
    supports __getitem__ and shape attributes, it will work.  If the data does not
    support __getitem__, the subclass MUST implement the `isel` method.  If the data
    does not have a `shape` attribute, the subclass MUST implement the `dims` and
    `coords` properties.
    """

    # Order in which subclasses are checked for support.
    # Lower numbers are checked first, and the first supporting subclass is used.
    # Default is 50, and fallback to numpy-like duckarray is 100.
    # Subclasses can override this to change the priority in which they are checked
    PRIORITY: ClassVar[int] = 50
    # These names will be checked when looking for a channel axis
    COMMON_CHANNEL_NAMES: ClassVar[Container[str]] = ("channel", "ch", "c")
    COMMON_Z_AXIS_NAMES: ClassVar[Container[str]] = ("z", "depth", "focus")

    # Maximum dimension size consider when guessing the channel axis
    MAX_CHANNELS: ClassVar[int] = 16

    dims_changed = Signal()
    """Signal to emit when the dimensions of the data change.

    NOTE: It is up to data wrappers, or even end-users to emit this signal when the
    dimensions/shape of the wrapped _data object changes.
    """
    data_changed = Signal()
    """Signal emitted when the data changes.

    NOTE: It is up to data wrappers, or even end-users to emit this signal when the
    data object changes.  We do not currently use object proxies to spy on mutation
    of the underlying data.
    """

    def __init__(self, data: ArrayT) -> None:
        self._data = data
        if not hasattr(self._data, "__getitem__") and "isel" not in type(self).__dict__:
            raise NotImplementedError(
                "DataWrapper subclass MUST implement `isel` method if data does not "
                "support __getitem__."
            )

        has_shape = hasattr(self._data, "shape") and isinstance(self._data.shape, tuple)
        has_methods = "dims" in type(self).__dict__ and "coords" in type(self).__dict__
        if not has_shape and not has_methods:
            raise NotImplementedError(
                "DataWrapper subclass MUST implement `dims` and `coords` properties"
                " if data does not have a `shape` attribute or if the shape is not "
                "a tuple."
            )
        self.dims_changed.connect(self.clear_cache)

    # ----------------------------- Mandatory methods -----------------------------

    @classmethod
    @abstractmethod
    def supports(cls, obj: Any) -> TypeGuard[Any]:
        """Return True if this wrapper can handle the given object.

        Any exceptions raised by this method will be suppressed, so it is safe to
        directly import necessary dependencies without a try/except block.
        """

    @property
    def dims(self) -> tuple[Hashable, ...]:
        """Return the dimension labels for the data."""
        # type ignore is asserted in the __init__ method
        return tuple(range(len(self._data.shape)))  # type: ignore [attr-defined]

    @property
    def coords(self) -> Mapping[Hashable, Sequence]:
        """Return the coordinates for the data."""
        dims = self.dims
        # type ignore is asserted in the __init__ method
        return {i: range(s) for i, s in zip(dims, self._data.shape)}  # type: ignore [attr-defined]

    def isel(self, index: Mapping[int, int | slice]) -> np.ndarray:
        """Return a slice of the data as a numpy array.

        `index` will look like (e.g.) `{0: slice(0, 10), 1: 5}`.
        The default implementation converts the index to a tuple of the same length as
        the self.dims, populating missing keys with `slice(None)`, and then slices the
        data array using __getitem__.
        """
        idx = tuple(index.get(k, slice(None)) for k in range(len(self.dims)))
        # this type ignore is asserted in the __init__ method
        # if the data does not support __getitem__, then the DataWrapper subclass will
        # fail to initialize
        return self._asarray(self._data[idx])  # type: ignore [index]

    def _asarray(self, data: Any) -> np.ndarray:
        """Convert data to a numpy array."""
        return np.asarray(data)

    def save_as_zarr(self, path: str) -> None:
        raise NotImplementedError("Saving as zarr is not supported for this data type")

    @property
    def dtype(self) -> np.dtype:
        """Return the dtype for the data."""
        try:
            return np.dtype(self._data.dtype)  # type: ignore[attr-defined]
        except AttributeError as e:
            raise NotImplementedError(
                "`dtype` property not properly implemented for DataWrapper of type: "
                f"{type(self)}"
            ) from e

    # -----------------------------

    @classmethod
    def create(cls, data: ArrayT) -> DataWrapper[ArrayT]:
        """Create a DataWrapper instance for the given data.

        This method will detect all subclasses of DataWrapper and check them in order of
        their `PRIORITY` class variable. The first subclass that
        [`supports`][ndv.DataWrapper.supports] the given data will be used to wrap it.

        !!! tip

            This means that you can subclass DataWrapper to handle new data types.
            Just make sure that your subclass is imported before calling `create`.

        If no subclasses support the data, a `NotImplementedError` is raised.

        If an instance of `DataWrapper` is passed in, it will be returned as-is.
        """
        if isinstance(data, DataWrapper):
            return data

        # check subclasses for support
        # This allows users to define their own DataWrapper subclasses which will
        # be automatically detected (assuming they have been imported by this point)
        for subclass in sorted(_recurse_subclasses(cls), key=lambda x: x.PRIORITY):
            try:
                if subclass.supports(data):
                    logging.debug(f"Using {subclass.__name__} to wrap {type(data)}")
                    return subclass(data)
            except Exception as e:
                warnings.warn(
                    f"Error checking DataWrapper subclass {subclass.__name__}: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        raise NotImplementedError(f"Don't know how to wrap type {type(data)}")

    @property
    def data(self) -> ArrayT:
        """Return the data being wrapped."""
        return self._data

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: type, handler: GetCoreSchemaHandler
    ) -> pydantic_core.CoreSchema:
        from pydantic_core import core_schema

        return core_schema.no_info_before_validator_function(
            function=cls.create,
            schema=core_schema.any_schema(),
        )

    def sizes(self) -> Mapping[Hashable, int]:
        """Return the sizes of the dimensions."""
        return {dim: len(self.coords[dim]) for dim in self.dims}

    # these guess_x methods may change in the future to become more agnostic to the
    # dimension name/semantics that they are guessing.

    def guess_channel_axis(self) -> Hashable | None:
        """Return the (best guess) axis name for the channel dimension."""
        # for arrays with labeled dimensions,
        # see if any of the dimensions are named "channel"
        sizes = self.sizes()
        if len(sizes) < 3 or min(sizes.values()) > self.MAX_CHANNELS:
            return None

        for dimkey, val in sizes.items():
            if str(dimkey).lower() in self.COMMON_CHANNEL_NAMES:
                if val <= self.MAX_CHANNELS:
                    return self.normalize_axis_key(dimkey)

        # otherwise use the smallest dimension as the channel axis
        return min(sizes, key=sizes.get)  # type: ignore [arg-type]

    def guess_z_axis(self) -> Hashable | None:
        """Return the (best guess) axis name for the z (3rd spatial) dimension."""
        sizes = self.sizes()
        ch = self.guess_channel_axis()
        for dimkey in sizes:
            if str(dimkey).lower() in self.COMMON_Z_AXIS_NAMES:
                if (normed := self.normalize_axis_key(dimkey)) != ch:
                    return normed

        # otherwise return the LAST axis that is neither in the last two dimensions
        # or the channel axis guess
        return next(
            (self.normalize_axis_key(x) for x in reversed(self.dims[:-2]) if x != ch),
            None,
        )

    def summary_info(self) -> str:
        """Return info label with information about the data."""
        package = getattr(self._data, "__module__", "").split(".")[0]
        info = f"{package}.{getattr(type(self._data), '__qualname__', '')}"

        if sizes := self.sizes():
            # if all of the dimension keys are just integers, omit them from size_str
            if all(isinstance(x, int) for x in sizes):
                size_str = repr(tuple(sizes.values()))
            # otherwise, include the keys in the size_str
            else:
                size_str = ", ".join(f"{k}:{v}" for k, v in sizes.items())
                size_str = f"({size_str})"
            info += f" {size_str}"
        if dtype := getattr(self._data, "dtype", ""):
            info += f", {dtype}"
        if nbytes := getattr(self._data, "nbytes", 0):
            info += f", {_human_readable_size(nbytes)}"
        return info

    @cached_property
    def axis_map(self) -> Mapping[Hashable, int]:
        """Mapping of ALL valid axis keys to normalized, positive integer keys."""
        axis_index: dict[Hashable, int] = {}
        ndims = len(self.dims)
        for i, dim in enumerate(self.dims):
            axis_index[dim] = i  # map dimension label to positive index
            axis_index[i] = i  # map positive integer index to itself
            axis_index[-(ndims - i)] = i  # map negative integer index to positive index
        return axis_index

    def normalize_axis_key(self, axis: Hashable) -> int:
        """Return positive index for `axis` (which can be +/- int or str label)."""
        try:
            return self.axis_map[axis]
        except KeyError as e:
            ndims = len(self.dims)
            if isinstance(axis, int):
                raise IndexError(
                    f"Axis index {axis} out of bounds for data with {ndims} dimensions"
                ) from e
            raise IndexError(f"Axis label {axis} not found in data dimensions") from e

    def clear_cache(self) -> None:
        """Clear any cached properties."""
        if hasattr(self, "axis_map"):
            del self.axis_map


def _human_readable_size(nbytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
    for unit in units:
        if nbytes < 1024:
            return f"{nbytes:.2f}".rstrip("0").rstrip(".") + unit
        nbytes /= 1024.0
    return f"{nbytes:.2f}YB"  # In case nbytes is extremely large


##########################


class ArrayLikeWrapper(DataWrapper[NPArrayLike]):
    """Wrapper for numpy duck array-like objects.

    The base class is suitable for any object that supports __getitem__ and shape.
    So we just need to implement supports to define the type of object we are wrapping.
    """

    PRIORITY = 100

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[NPArrayLike]:
        if (
            (
                isinstance(obj, np.ndarray)
                or hasattr(obj, "__array_function__")
                or hasattr(obj, "__array_namespace__")
                or hasattr(obj, "__array__")
            )
            and hasattr(obj, "__getitem__")
            and hasattr(obj, "shape")
        ):
            return True
        return False


class DaskWrapper(DataWrapper["da.Array"]):
    """Wrapper for dask array objects."""

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[da.Array]:
        if (da := sys.modules.get("dask.array")) and isinstance(obj, da.Array):
            return True
        return False

    def _asarray(self, data: da.Array) -> np.ndarray:
        return np.asarray(data.compute())

    def save_as_zarr(self, path: str) -> None:
        self._data.to_zarr(url=path)


class SparseArrayWrapper(DataWrapper["sparse.Array"]):
    PRIORITY = 50

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[sparse.COO]:
        if (sparse := sys.modules.get("sparse")) and isinstance(obj, sparse.COO):
            return True
        return False

    def _asarray(self, data: sparse.COO) -> np.ndarray:
        return np.asarray(data.todense())


class CLArrayWrapper(DataWrapper["cl_array.Array"]):
    """Wrapper for pyopencl array objects."""

    PRIORITY = 50

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[cl_array.Array]:
        if (cl_array := sys.modules.get("pyopencl.array")) and isinstance(
            obj, cl_array.Array
        ):
            return True
        return False

    def _asarray(self, data: cl_array.Array) -> np.ndarray:
        return np.asarray(data.get())


class XarrayWrapper(DataWrapper["xr.DataArray"]):
    """Wrapper for xarray DataArray objects."""

    @property
    def dims(self) -> tuple[Hashable, ...]:
        """Return the dimension labels for the data."""
        return tuple(self._data.dims)

    @property
    def coords(self) -> Mapping[Hashable, Sequence]:
        """Return the coordinates for the data."""
        return self.data.coords  # type: ignore [no-any-return]

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[xr.DataArray]:
        if (xr := sys.modules.get("xarray")) and isinstance(obj, xr.DataArray):
            return True
        return False


class TensorstoreWrapper(DataWrapper["ts.TensorStore"]):
    """Wrapper for tensorstore.TensorStore objects."""

    def __init__(self, data: Any) -> None:
        super().__init__(data)

        import tensorstore as ts

        self._ts = ts

        spec = self.data.spec().to_json()
        dims: Sequence[Hashable] | None = None
        self._ts = ts
        if (tform := spec.get("transform")) and ("input_labels" in tform):
            dims = [str(x) for x in tform["input_labels"]]
        elif (
            str(spec.get("driver")).startswith("zarr")
            and (zattrs := self.data.kvstore.read(".zattrs").result().value)
            and isinstance((zattr_dict := json.loads(zattrs)), dict)
            and "_ARRAY_DIMENSIONS" in zattr_dict
        ):
            dims = zattr_dict["_ARRAY_DIMENSIONS"]

        if isinstance(dims, Sequence) and len(dims) == len(self._data.domain):
            self._dims: tuple[Hashable, ...] = tuple(str(x) for x in dims)
            self._data = self.data[ts.d[:].label[self._dims]]
        else:
            self._dims = tuple(range(len(self._data.domain)))
        self._coords: Mapping[Hashable, Sequence] = {
            i: range(s) for i, s in zip(self._dims, self._data.domain.shape)
        }

    @property
    def dtype(self) -> np.dtype:
        """Return the dtype for the data."""
        return np.dtype(str(self._data.dtype.name))

    @property
    def dims(self) -> tuple[Hashable, ...]:
        """Return the dimension labels for the data."""
        return self._dims

    @property
    def coords(self) -> Mapping[Hashable, Sequence]:
        """Return the coordinates for the data."""
        return self._coords

    def sizes(self) -> Mapping[Hashable, int]:
        return dict(zip(self._dims, self._data.domain.shape))

    def isel(self, indexers: Mapping[int, int | slice]) -> np.ndarray:
        if not indexers:
            slc: slice | tuple = slice(None)
        else:
            slc = tuple(
                indexers.get(i, slice(None)) for i in range(len(self._data.shape))
            )
        result = self._data[slc].read().result()
        return np.asarray(result)

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[ts.TensorStore]:
        if (ts := sys.modules.get("tensorstore")) and isinstance(obj, ts.TensorStore):
            return True
        return False


class TorchTensorWrapper(DataWrapper["torch.Tensor"]):
    """Wrapper for torch tensor objects."""

    @property
    def coords(self) -> Mapping[Hashable, Sequence]:
        dims = self.dims
        return {i: range(s) for i, s in zip(dims, self.data.shape)}

    @property
    def dims(self) -> tuple[Hashable, ...]:
        # torch does enforce that len(names) == len(shape), and that names are unique
        # with the exception of None, which is allowed to be repeated
        if hasattr(self.data, "names"):
            names = self.data.names
            return tuple((i if name is None else name) for i, name in enumerate(names))
        return tuple(range(len(self.data.shape)))

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[torch.Tensor]:
        if (torch := sys.modules.get("torch")) and isinstance(obj, torch.Tensor):
            return True
        return False


class RingBufferWrapper(DataWrapper[RingBuffer]):
    """Wrapper for ring buffer objects."""

    def __init__(
        self,
        max_capacity: int | RingBuffer,
        dtype: npt.DTypeLike = None,
        *,
        allow_overwrite: bool = True,
    ):
        if isinstance(max_capacity, RingBuffer):
            if dtype is not None:  # pragma: no cover
                raise ValueError(
                    "Cannot specify dtype when passing an existing RingBuffer."
                )
            self._ring = max_capacity
        else:
            if dtype is None:
                dtype = float
            self._ring = RingBuffer(
                max_capacity=max_capacity, dtype=dtype, allow_overwrite=allow_overwrite
            )
        self._ring.resized.connect(self.dims_changed)
        super().__init__(self._ring)

    @property
    def dims(self) -> tuple[int, ...]:
        """Return the dimensions of the data."""
        return tuple(range(len(self._ring.shape)))

    @property
    def coords(self) -> Mapping:
        """Return the coordinates for the data."""
        shape = self._ring.shape
        return {i: range(s) for i, s in enumerate(shape)}

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[np.ndarray]:
        if isinstance(obj, RingBuffer):
            return True
        return False

    def append(self, value: npt.ArrayLike) -> None:
        """Append a value to the right end of the buffer."""
        self._ring.append(value)

    def appendleft(self, value: npt.ArrayLike) -> None:
        """Append a value to the left end of the buffer."""
        self._ring.appendleft(value)

    def pop(self) -> np.ndarray:
        """Pop a value from the right end of the buffer."""
        return self._ring.pop()

    def popleft(self) -> np.ndarray:
        """Pop a value from the left end of the buffer."""
        return self._ring.popleft()


class NGFFZarrWrapper(DataWrapper[Any]):
    """Wrapper for NGFF-Zarr (OME-Zarr) objects.

    Supports yaozarrs.ZarrGroup, zarr.Group, and string paths to zarr stores.
    All inputs are converted to yaozarrs.ZarrGroup for consistent handling.

    Handles both Image and Plate (HCS) datasets by navigating to the first
    available image array.
    """

    PRIORITY = 40  # Higher priority than generic array-like

    def __init__(self, data: Any) -> None:
        """Initialize the wrapper with NGFF-Zarr data.

        Parameters
        ----------
        data : yaozarrs.ZarrGroup | zarr.Group | str | os.PathLike
            The NGFF-Zarr data to wrap. Can be:
            - A yaozarrs.ZarrGroup
            - A zarr.Group (will be converted to yaozarrs.ZarrGroup)
            - A string/path to a zarr store (will be opened with yaozarrs)
        """
        import os

        try:
            import yaozarrs as yaz
        except ImportError as e:
            raise ImportError(
                "yaozarrs is required to use NGFFZarrWrapper. "
                "Please install it via 'pip install yaozarrs'."
            ) from e

        # Convert all inputs to yaozarrs.ZarrGroup
        if isinstance(data, (str, os.PathLike)):
            self._zarr_group = yaz.open_group(data)
        elif isinstance(data, yaz._zarr.ZarrGroup):
            # Already a yaozarrs.ZarrGroup
            self._zarr_group = data
        elif hasattr(data, "store") and hasattr(data, "info"):
            # Handle zarr.Group by opening with yaozarrs using store path
            # zarr v3 uses store.root, v2 uses store.path
            store_path = getattr(data.store, "root", getattr(data.store, "path", None))
            if store_path is None:
                raise ValueError("Could not determine zarr store path")
            self._zarr_group = yaz.open_group(store_path)
        else:
            raise ValueError(
                f"Unsupported data type: {type(data)}. Expected yaozarrs.ZarrGroup, "
                "zarr.Group, or str/PathLike"
            )

        # Get the OME metadata
        self._ome_metadata = self._zarr_group.ome_metadata()

        if self._ome_metadata is None:
            raise ValueError("No OME metadata found in zarr group")

        # Navigate to the image based on metadata type
        self._setup_from_metadata()

        # Store the array as _data for compatibility with DataWrapper
        super().__init__(self._array)

    def _setup_from_metadata(self) -> None:
        """Setup axes and array based on the OME metadata type."""
        # Check if it's a Plate (HCS) dataset
        if hasattr(self._ome_metadata, "plate"):
            self._setup_from_plate()
        # Check if it's a Well dataset
        elif hasattr(self._ome_metadata, "well"):
            self._setup_from_well()
        # Check if this is an Image at the root, but has a Plate one level down
        # (common NGFF HCS structure: root/plate_name/wells/...)
        elif hasattr(self._ome_metadata, "multiscales"):
            # Check if any child groups have plate metadata
            plate_group = self._find_plate_in_children()
            if plate_group is not None:
                # Use the plate group instead
                self._zarr_group = plate_group
                self._ome_metadata = plate_group.ome_metadata()
                self._setup_from_plate()
            else:
                # Regular image
                self._setup_from_image(self._zarr_group)
        else:
            raise ValueError(
                f"Unsupported OME metadata type: {type(self._ome_metadata).__name__}"
            )

    def _find_plate_in_children(self) -> Any:
        """Check if any child groups contain plate metadata.

        Returns the first child group with plate metadata, or None.
        """
        from urllib.parse import unquote

        from yaozarrs._zarr import ZarrGroup

        try:
            # Get children from the group
            # yaozarrs doesn't have a direct way to list children, so we'll
            # try to access the store to get child names
            if hasattr(self._zarr_group, "store_path"):
                import os

                store_path = self._zarr_group.store_path
                # Handle file:// URI
                if store_path.startswith("file://"):
                    store_path = unquote(store_path[7:])

                if os.path.isdir(store_path):
                    # List directories (potential child groups)
                    children = [
                        d
                        for d in os.listdir(store_path)
                        if os.path.isdir(os.path.join(store_path, d))
                        and not d.startswith(".")
                    ]

                    # Check each child for plate metadata
                    for child_name in children:
                        try:
                            child_group = self._zarr_group[child_name]
                            if isinstance(child_group, ZarrGroup):
                                child_meta = child_group.ome_metadata()
                                if hasattr(child_meta, "plate"):
                                    return child_group
                        except Exception:
                            continue
        except Exception:
            pass

        return None

    def _setup_from_plate(self) -> None:
        """Setup from Plate (HCS) dataset: collect all wells/fields as positions."""
        from yaozarrs._zarr import ZarrArray, ZarrGroup

        plate = self._ome_metadata.plate
        if not plate.wells:
            raise ValueError("Plate metadata has no wells")

        # Collect all field arrays across all wells
        self._position_arrays = []
        self._position_info = []  # Store (well_path, field_path) for debugging

        for well_info in plate.wells:
            well_node = self._zarr_group[well_info.path]
            if not isinstance(well_node, ZarrGroup):
                continue

            well_metadata = well_node.ome_metadata()
            if not hasattr(well_metadata, "well") or not well_metadata.well.images:
                continue

            # Get all fields in this well
            for field_info in well_metadata.well.images:
                field_node = well_node[field_info.path]

                # The field can be either a ZarrArray directly or a ZarrGroup
                if isinstance(field_node, ZarrArray):
                    # Direct array case
                    self._position_arrays.append(field_node)
                    self._position_info.append((well_info.path, field_info.path))
                elif isinstance(field_node, ZarrGroup):
                    # Group with multiscales case
                    field_metadata = field_node.ome_metadata()
                    if not hasattr(field_metadata, "multiscales"):
                        continue

                    multiscale = field_metadata.multiscales[0]
                    dataset_path = multiscale.datasets[0].path
                    array = field_node[dataset_path]

                    self._position_arrays.append(array)
                    self._position_info.append((well_info.path, field_info.path))

        if not self._position_arrays:
            raise ValueError("No valid field arrays found in plate")

        # Get axes - need to determine from the first array's metadata
        first_array = self._position_arrays[0]
        if hasattr(first_array, "_metadata") and hasattr(
            first_array._metadata, "dimension_names"
        ):
            image_axes = list(first_array._metadata.dimension_names)
        else:
            # Fallback: assume standard NGFF axes based on shape
            shape = (
                first_array._metadata.shape
                if hasattr(first_array, "_metadata")
                else (1, 1, 512, 512)
            )
            ndim = len(shape)
            # Common pattern: t, c, z, y, x (depending on ndim)
            if ndim == 5:
                image_axes = ["t", "c", "z", "y", "x"]
            elif ndim == 4:
                image_axes = ["t", "c", "y", "x"]
            elif ndim == 3:
                image_axes = ["c", "y", "x"]
            elif ndim == 2:
                image_axes = ["y", "x"]
            else:
                image_axes = [f"d{i}" for i in range(ndim)]

        # Add position dimension at the beginning
        self._axes = ["p", *image_axes]
        self._array = None  # We handle this specially in isel()

    def _setup_from_well(self) -> None:
        """Setup from Well dataset: collect all fields as positions."""
        from yaozarrs._zarr import ZarrGroup

        well = self._ome_metadata.well
        if not well.images:
            raise ValueError("Well metadata has no images")

        # Collect all field arrays
        self._position_arrays = []
        self._position_info = []

        for field_info in well.images:
            field_node = self._zarr_group[field_info.path]
            if not isinstance(field_node, ZarrGroup):
                continue

            # Get the array from this field
            field_metadata = field_node.ome_metadata()
            if not hasattr(field_metadata, "multiscales"):
                continue

            multiscale = field_metadata.multiscales[0]
            dataset_path = multiscale.datasets[0].path
            array = field_node[dataset_path]

            self._position_arrays.append(array)
            self._position_info.append(("", field_info.path))

        if not self._position_arrays:
            raise ValueError("No valid field arrays found in well")

        # Get axes from the first field
        first_field_group = self._zarr_group[self._position_info[0][1]]
        first_field_meta = first_field_group.ome_metadata()
        multiscale = first_field_meta.multiscales[0]
        image_axes = [axis.name for axis in multiscale.axes]

        # Add position dimension at the beginning
        self._axes = ["p", *image_axes]
        self._array = None  # We handle this specially in isel()

    def _setup_from_image(self, image_group: Any) -> None:
        """Setup axes and array from an Image group.

        Parameters
        ----------
        image_group : yaozarrs.ZarrGroup
            The zarr group containing image metadata
        """
        # Get image metadata
        image_metadata = image_group.ome_metadata()
        if not hasattr(image_metadata, "multiscales"):
            raise ValueError("Image metadata has no multiscales")

        # Extract axes and dataset information from the first multiscale
        if not image_metadata.multiscales:
            raise ValueError("No multiscales found in image metadata")

        multiscale = image_metadata.multiscales[0]
        self._axes = [axis.name for axis in multiscale.axes]

        # Get the first (highest resolution) dataset
        if not multiscale.datasets:
            raise ValueError("No datasets found in multiscale metadata")

        self._dataset_path = multiscale.datasets[0].path
        # Access the array
        self._array = image_group[self._dataset_path]

    @property
    def dims(self) -> tuple[Hashable, ...]:
        """Return the dimension labels for the data."""
        return tuple(self._axes)

    @property
    def coords(self) -> Mapping[Hashable, Sequence]:
        """Return the coordinates for the data."""
        # Handle HCS datasets with multiple positions
        if hasattr(self, "_position_arrays") and self._position_arrays:
            # Get shape from first position array
            first_array = self._position_arrays[0]
            if hasattr(first_array, "_metadata") and hasattr(
                first_array._metadata, "shape"
            ):
                shape = first_array._metadata.shape
            else:
                raise ValueError("Could not determine array shape")

            # Build coords with position dimension first
            coords = {"p": range(len(self._position_arrays))}
            coords.update(
                {axis: range(size) for axis, size in zip(self._axes[1:], shape)}
            )
            return coords

        # Single image case
        if hasattr(self._array, "_metadata") and hasattr(
            self._array._metadata, "shape"
        ):
            shape = self._array._metadata.shape
        else:
            raise ValueError("Could not determine array shape")

        return {axis: range(size) for axis, size in zip(self._axes, shape)}

    @property
    def dtype(self) -> np.dtype:
        """Return the dtype for the data."""
        if hasattr(self._array, "dtype"):
            return np.dtype(self._array.dtype)
        raise ValueError("Could not determine array dtype")

    def isel(self, index: Mapping[int, int | slice]) -> np.ndarray:
        """Return a slice of the data as a numpy array.

        Parameters
        ----------
        index : Mapping[int, int | slice]
            Mapping of axis indices to slice/index values
        """
        # Handle HCS datasets with multiple positions
        if hasattr(self, "_position_arrays") and self._position_arrays:
            # Extract position index (dimension 0)
            pos_idx = index.get(0, 0)

            # Handle slicing across positions
            if isinstance(pos_idx, slice):
                # Determine which positions to include
                num_positions = len(self._position_arrays)
                positions = range(num_positions)[pos_idx]

                if not positions:
                    # Empty slice
                    return np.array([])

                # Build index for the remaining dimensions (shift by 1)
                image_index = {k - 1: v for k, v in index.items() if k > 0}

                # Collect data from each position
                slices = []
                for p in positions:
                    array = self._position_arrays[p]
                    try:
                        zarr_array = array.to_zarr_python()
                        idx = tuple(
                            image_index.get(k, slice(None))
                            for k in range(len(self._axes) - 1)
                        )
                        slices.append(np.asarray(zarr_array[idx]))
                    except ImportError:
                        ts_array = array.to_tensorstore()
                        idx = tuple(
                            image_index.get(k, slice(None))
                            for k in range(len(self._axes) - 1)
                        )
                        result = ts_array[idx].read().result()
                        slices.append(np.asarray(result))

                # Stack along first axis
                return np.stack(slices, axis=0)

            # Get the array for this position
            array = self._position_arrays[pos_idx]

            # Build index for the remaining dimensions (shift by 1)
            image_index = {k - 1: v for k, v in index.items() if k > 0}

            # Access the data from this position's array
            try:
                zarr_array = array.to_zarr_python()
                idx = tuple(
                    image_index.get(k, slice(None)) for k in range(len(self._axes) - 1)
                )
                return np.asarray(zarr_array[idx])
            except ImportError:
                try:
                    ts_array = array.to_tensorstore()
                    idx = tuple(
                        image_index.get(k, slice(None))
                        for k in range(len(self._axes) - 1)
                    )
                    result = ts_array[idx].read().result()
                    return np.asarray(result)
                except ImportError as e:
                    msg = (
                        "Either 'zarr' or 'tensorstore' package is required "
                        "for data access. Install with: pip install zarr or "
                        "pip install tensorstore"
                    )
                    raise ImportError(msg) from e

        # Single image case
        try:
            zarr_array = self._array.to_zarr_python()
            idx = tuple(index.get(k, slice(None)) for k in range(len(self._axes)))
            return np.asarray(zarr_array[idx])
        except ImportError:
            try:
                ts_array = self._array.to_tensorstore()
                idx = tuple(index.get(k, slice(None)) for k in range(len(self._axes)))
                result = ts_array[idx].read().result()
                return np.asarray(result)
            except ImportError as e:
                msg = (
                    "Either 'zarr' or 'tensorstore' package is required "
                    "for data access. Install with: pip install zarr or "
                    "pip install tensorstore"
                )
                raise ImportError(msg) from e

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[Any]:
        """Return True if this wrapper can handle the given object.

        Supports:
        - yaozarrs.ZarrGroup objects
        - zarr.Group objects
        - String/path to zarr stores (with valid NGFF metadata)
        """
        import os

        try:
            import yaozarrs
        except ImportError:
            warnings.warn(
                "yaozarrs is not installed; NGFFZarrWrapper cannot be used.",
                stacklevel=2,
            )
            return False

        # Check for string/path
        if isinstance(obj, (str, os.PathLike)):
            if not str(obj).endswith(".zarr"):
                return False
            try:
                # First validate the store
                yaozarrs.validate_zarr_store(obj)
                # Then check for OME metadata
                group = yaozarrs.open_group(obj)
                return group.ome_metadata() is not None
            except Exception:
                return False

        # Check for yaozarrs.ZarrGroup
        if isinstance(obj, yaozarrs._zarr.ZarrGroup):
            # Check if it has OME metadata
            try:
                return obj.ome_metadata() is not None
            except Exception:
                return False

        # Check for zarr.Group
        try:
            import zarr
        except ImportError:
            return False

        if isinstance(obj, zarr.Group):
            try:
                # zarr v3 uses store.root, v2 uses store.path
                store_path = getattr(
                    obj.store, "root", getattr(obj.store, "path", None)
                )
                if store_path is None:
                    return False
                group = yaozarrs.open_group(store_path)
                return group.ome_metadata() is not None
            except Exception:
                return False

        return False

    def save_as_zarr(self, path: str) -> None:
        """Save the NGFF-Zarr data to a new location.

        Note: This creates a copy of the entire zarr store.
        """
        import shutil
        from pathlib import Path

        # Get the source path from the zarr group
        source_path = self._zarr_group.store_path
        dest_path = Path(path)

        if dest_path.exists():
            raise FileExistsError(f"Destination path already exists: {path}")

        # Copy the entire zarr store
        shutil.copytree(source_path, dest_path)
