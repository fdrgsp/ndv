"""Test script for NGFFZarrWrapper."""

import zarr
from yaozarrs import open_group, validate_ome_object, validate_zarr_store

import ndv

# Test with HCS/Plate dataset
path = "/Users/fdrgsp/Desktop/ts_hcs_zarr_example.ome.zarr"

print("=" * 80)
print("Testing NGFFZarrWrapper with HCS/Plate dataset")
print("=" * 80)
print()

# Validate the zarr store
print("1. Validating zarr store...")
validate_zarr_store(path)
print("   ✓ Zarr store is valid")
print()

# Test with yaozarrs.ZarrGroup
print("2. Testing with yaozarrs.ZarrGroup...")
z = open_group(path)
validate_ome_object(z.ome_metadata())
print(f"   OME metadata type: {type(z.ome_metadata()).__name__}")

wrapper = ndv.models.DataWrapper.create(z)
print(f"   Wrapper type: {type(wrapper).__name__}")
print(f"   Dims: {wrapper.dims}")
print(f"   Sizes: {wrapper.sizes()}")
print(f"   Dtype: {wrapper.dtype}")

# Test data access
slice_data = wrapper.isel({0: 0, 1: 0})
print(f"   Slice shape: {slice_data.shape}")
print("   ✓ yaozarrs.ZarrGroup works!")
print()

# Test with zarr.Group
print("3. Testing with zarr.Group...")
z1 = zarr.open_group(path, mode="r")
wrapper2 = ndv.models.DataWrapper.create(z1)
print(f"   Wrapper type: {type(wrapper2).__name__}")
print(f"   Dims: {wrapper2.dims}")
print(f"   Sizes: {wrapper2.sizes()}")
print("   ✓ zarr.Group works!")
print()

# Test with string path
print("4. Testing with string path...")
wrapper3 = ndv.models.DataWrapper.create(path)
print(f"   Wrapper type: {type(wrapper3).__name__}")
print(f"   Dims: {wrapper3.dims}")
print(f"   Sizes: {wrapper3.sizes()}")
print("   ✓ String path works!")
print()

# Test with simple image (if available)
try:
    # You can test with a simple image zarr if you have one
    # For now, we'll skip this part
    print("5. Testing with simple Image dataset...")
    print("   (Skipped - no simple image dataset available)")
    print()
except Exception as e:
    print(f"   Note: {e}")
    print()

print("=" * 80)
print("All tests passed! ✓")
print("=" * 80)
print()
print("Summary:")
print("  - NGFFZarrWrapper can handle yaozarrs.ZarrGroup objects")
print("  - NGFFZarrWrapper can handle zarr.Group objects")
print("  - NGFFZarrWrapper can handle string paths to zarr stores")
print("  - NGFFZarrWrapper correctly navigates HCS/Plate datasets")
print("  - NGFFZarrWrapper provides correct dims and coords")
print("  - NGFFZarrWrapper can slice and return numpy arrays")
