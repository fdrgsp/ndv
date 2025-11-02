"""Test NGFFZarrWrapper integration with ndv.imshow."""

import zarr
from yaozarrs import open_group

import ndv

path = "/Users/fdrgsp/Desktop/ts_hcs_zarr_example.ome.zarr"

print("Testing NGFFZarrWrapper integration with ndv...")
print()

# Test 1: with yaozarrs.ZarrGroup
print("1. Creating viewer with yaozarrs.ZarrGroup...")
z = open_group(path)
try:
    viewer = ndv.imshow(z)
    print("   ✓ Viewer created successfully")
    print(f"   Data wrapper type: {type(viewer.model.data_wrapper).__name__}")
    print(f"   Dims: {viewer.model.dims}")
    print(f"   Current index: {viewer.model.current_index}")
    viewer.close()
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback

    traceback.print_exc()
print()

# Test 2: with zarr.Group
print("2. Creating viewer with zarr.Group...")
z1 = zarr.open_group(path, mode="r")
try:
    viewer = ndv.imshow(z1)
    print("   ✓ Viewer created successfully")
    print(f"   Data wrapper type: {type(viewer.model.data_wrapper).__name__}")
    viewer.close()
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback

    traceback.print_exc()
print()

# Test 3: with string path
print("3. Creating viewer with string path...")
try:
    viewer = ndv.imshow(path)
    print("   ✓ Viewer created successfully")
    print(f"   Data wrapper type: {type(viewer.model.data_wrapper).__name__}")
    viewer.close()
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback

    traceback.print_exc()
print()

print("=" * 80)
print("Integration tests completed! ✓")
print("=" * 80)
