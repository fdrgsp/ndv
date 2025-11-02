# NGFFZarrWrapper - DataWrapper for OME-NGFF Zarr Files

## Overview

The `NGFFZarrWrapper` is a new `DataWrapper` subclass designed to handle NGFF (Next Generation File Format) Zarr files, also known as OME-Zarr files. It uses the `yaozarrs` library for metadata parsing and navigation.

## Features

### Supported Input Types

The wrapper accepts three types of input and converts them all to `yaozarrs.ZarrGroup` internally:

1. **yaozarrs.ZarrGroup** - Direct yaozarrs group objects
2. **zarr.Group** - Standard zarr-python group objects (converted to yaozarrs)
3. **String/Path** - File paths to zarr stores (opened with yaozarrs)

### Supported NGFF Structures

The wrapper intelligently navigates different NGFF metadata types:

1. **Image** - Simple image datasets with multiscale pyramids
2. **Plate** - High Content Screening (HCS) datasets with wells and fields
3. **Well** - Individual well datasets with multiple fields

The wrapper automatically navigates to the first available image array regardless of the structure type.

## Implementation Details

### Key Components

1. **Metadata Handling**
   - Uses `yaozarrs` for parsing OME-NGFF v0.4 and v0.5 metadata
   - Extracts axis information from multiscale metadata
   - Supports labeled axes (e.g., 't', 'c', 'z', 'y', 'x')

2. **Data Access**
   - Uses the first (highest resolution) dataset from multiscale pyramids
   - Supports data access via either zarr-python or tensorstore
   - Returns numpy arrays from `isel()` method

3. **Dimension Support**
   - Provides proper dimension labels from OME metadata
   - Returns coordinates as ranges based on array shape
   - Supports the full axis naming convention from NGFF spec

### Code Location

The implementation is in `/Users/fdrgsp/Documents/git/ndv/src/ndv/models/_data_wrapper.py`

### Priority

- `PRIORITY = 40` (higher than generic array-like wrappers at 50, lower than specialized wrappers)

## Usage Examples

### Example 1: Opening with yaozarrs

```python
from yaozarrs import open_group
import ndv

# Open with yaozarrs
z = open_group("/path/to/data.ome.zarr")
ndv.imshow(z)
```

### Example 2: Opening with zarr-python

```python
import zarr
import ndv

# Open with standard zarr
z = zarr.open_group("/path/to/data.ome.zarr", mode="r")
ndv.imshow(z)
```

### Example 3: Opening from path

```python
import ndv

# Open directly from path
ndv.imshow("/path/to/data.ome.zarr")
```

### Example 4: Working with HCS/Plate data

```python
import ndv

# Automatically navigates to first well/field
ndv.imshow("/path/to/plate.ome.zarr")
```

## Testing

Comprehensive tests have been created:

- `test_ngff_wrapper.py` - Unit tests for the wrapper functionality
- `test_ngff_integration.py` - Integration tests with ndv.imshow

All tests pass successfully with:
- yaozarrs.ZarrGroup objects
- zarr.Group objects  
- String paths to zarr stores
- HCS/Plate datasets

## Dependencies

### Required
- `yaozarrs` - For OME-NGFF metadata parsing and zarr navigation
- `fsspec` - Required by yaozarrs for filesystem operations

### Optional (for data access)
- `zarr` - For reading zarr arrays (preferred)
- `tensorstore` - Alternative for reading zarr arrays

If neither zarr nor tensorstore is available, an informative error message is shown.

## Advantages of Using yaozarrs

1. **Minimal dependencies** - yaozarrs only requires pydantic
2. **Schema validation** - Built-in validation of NGFF metadata
3. **Version agnostic** - Supports both NGFF v0.4 and v0.5
4. **I/O flexibility** - Doesn't force a specific zarr implementation
5. **Metadata-first** - Clean separation of metadata from data access

## Future Enhancements

Potential improvements for the future:

1. Support for selecting specific wells/fields in Plate datasets
2. Support for Labels/segmentation datasets
3. Caching of multiscale levels for faster zoom/pan
4. Integration with OMERO metadata for channel colors/names
5. Support for writing NGFF-Zarr files

## Notes

- The wrapper automatically selects the first (highest resolution) array from multiscale pyramids
- For Plate datasets, it navigates to the first well and first field
- All dimension labels come from the OME metadata, ensuring consistency
- Data slicing returns numpy arrays via either zarr-python or tensorstore
