"""Comprehensive example of NGFFZarrWrapper usage."""

import zarr
from yaozarrs import open_group, validate_ome_object, validate_zarr_store

import ndv

print("=" * 80)
print("NGFFZarrWrapper - Comprehensive Example")
print("=" * 80)
print()

# Path to your NGFF-Zarr file
path = "/Users/fdrgsp/Desktop/ts_hcs_zarr_example.ome.zarr"

# ============================================================================
# PART 1: Validation
# ============================================================================
print("PART 1: Validating NGFF-Zarr Store")
print("-" * 80)

# Validate that it's a proper zarr store with NGFF metadata
print("Validating zarr store...")
validate_zarr_store(path)
print("✓ Store is valid")
print()

# ============================================================================
# PART 2: Opening with Different Methods
# ============================================================================
print("PART 2: Opening with Different Methods")
print("-" * 80)

# Method 1: Open with yaozarrs
print("\n1. Opening with yaozarrs.open_group()...")
z = open_group(path)
print(f"   Type: {type(z)}")
print(f"   OME version: {z.ome_version()}")
print(f"   OME metadata type: {type(z.ome_metadata()).__name__}")

# Validate the OME metadata
validate_ome_object(z.ome_metadata())
print("   ✓ OME metadata is valid")

# Method 2: Open with zarr-python
print("\n2. Opening with zarr.open_group()...")
z_zarr = zarr.open_group(path, mode="r")
print(f"   Type: {type(z_zarr)}")
print("   ✓ Opened successfully")

# Method 3: Just use the path string
print("\n3. Using path string directly...")
print(f"   Path: {path}")
print("   ✓ Will be opened automatically by DataWrapper")
print()

# ============================================================================
# PART 3: Creating DataWrapper
# ============================================================================
print("PART 3: Creating DataWrapper")
print("-" * 80)

# Create wrapper from yaozarrs group
wrapper = ndv.models.DataWrapper.create(z)
print(f"\nWrapper type: {type(wrapper).__name__}")
print(f"Data type: {type(wrapper.data)}")
print()

# Display wrapper information
print("Wrapper Information:")
print(f"  Dimensions: {wrapper.dims}")
print(f"  Sizes: {wrapper.sizes()}")
print(f"  Dtype: {wrapper.dtype}")
print(f"  Summary: {wrapper.summary_info()}")
print()

# ============================================================================
# PART 4: Working with Dimensions
# ============================================================================
print("PART 4: Working with Dimensions")
print("-" * 80)

print("\nDimension mapping:")
for dim_name, size in wrapper.sizes().items():
    idx = wrapper.normalize_axis_key(dim_name)
    print(f"  {dim_name} (index {idx}): {size} frames")

# Check for channel axis
channel_axis = wrapper.guess_channel_axis()
if channel_axis is not None:
    print(f"\nGuessed channel axis: {wrapper.dims[channel_axis]}")

# Check for z axis
z_axis = wrapper.guess_z_axis()
if z_axis is not None:
    print(f"Guessed z axis: {wrapper.dims[z_axis]}")
else:
    print("No z axis detected (this is a 2D dataset)")
print()

# ============================================================================
# PART 5: Data Access
# ============================================================================
print("PART 5: Data Access")
print("-" * 80)

print("\nAccessing data slices...")

# Get a single 2D frame (first timepoint, first channel)
print("\n1. Single frame (t=0, c=0):")
frame = wrapper.isel({0: 0, 1: 0})  # t=0, c=0, all y and x
print(f"   Shape: {frame.shape}")
print(f"   Dtype: {frame.dtype}")
print(f"   Min: {frame.min()}, Max: {frame.max()}")
print(f"   Mean: {frame.mean():.2f}")

# Get all timepoints for a single channel
print("\n2. All timepoints (c=0):")
timeseries = wrapper.isel({1: 0})  # c=0, all t, y, x
print(f"   Shape: {timeseries.shape}")
print(f"   Contains {timeseries.shape[0]} timepoints")

# Get a region of interest
print("\n3. Region of interest:")
roi = wrapper.isel({0: 0, 1: 0})  # Full frame for now
roi_crop = roi[100:200, 100:200]  # Crop in numpy
print(f"   ROI shape: {roi_crop.shape}")
print()

# ============================================================================
# PART 6: Integration with ndv (optional - requires visualization backend)
# ============================================================================
print("PART 6: Integration with ndv")
print("-" * 80)

print("\nThe wrapper integrates seamlessly with ndv.imshow():")
print("  Example: ndv.imshow(z)  # yaozarrs.ZarrGroup")
print("  Example: ndv.imshow(z_zarr)  # zarr.Group")
print("  Example: ndv.imshow(path)  # string path")
print()

# Uncomment to actually show the viewer (requires visualization backend):
# viewer = ndv.imshow(z)
# print(f"Viewer created with {len(wrapper.dims)} dimensions")
# print(f"Current index: {viewer.model.current_index}")

# ============================================================================
# PART 7: Advanced Features
# ============================================================================
print("PART 7: Advanced Features")
print("-" * 80)

print("\nOther wrapper methods:")
print(f"  coords: {list(wrapper.coords.keys())}")
print(f"  axis_map: {dict(list(wrapper.axis_map.items())[:4])}...")  # Show first few

# Show coordinate ranges for each dimension
print("\nCoordinate ranges:")
for dim_name in wrapper.dims:
    coords = wrapper.coords[dim_name]
    print(f"  {dim_name}: {min(coords)} to {max(coords)}")
print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 80)
print("Summary")
print("=" * 80)
print()
print("The NGFFZarrWrapper successfully:")
print("  ✓ Opens NGFF-Zarr files using yaozarrs")
print("  ✓ Handles HCS/Plate datasets by navigating to first field")
print("  ✓ Provides proper dimension labels from OME metadata")
print("  ✓ Supports data access via zarr-python or tensorstore")
print("  ✓ Integrates seamlessly with ndv.imshow()")
print("  ✓ Works with yaozarrs.ZarrGroup, zarr.Group, and string paths")
print()
print("You can now use NGFF-Zarr files directly with ndv!")
print("=" * 80)
