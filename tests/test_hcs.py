#!/usr/bin/env python3
"""Test HCS support in NDV."""

import yaozarrs
import ndv

# Open the HCS plate
path = '/Users/fdrgsp/Desktop/acqz_hcs_zarr_example.ome.zarr/96-well'
z = yaozarrs.open_group(path)

print("Opening HCS dataset...")
print(f"Type: {type(z)}")
print(f"Path: {path}")

# Create wrapper
wrapper = ndv.models.DataWrapper.create(z)
print(f"\nWrapper dims: {wrapper.dims}")
print(f"Wrapper sizes: {wrapper.sizes()}")

print("\nPosition info:")
for i, (well, field) in enumerate(wrapper._position_info):
    print(f"  [{i}] {well}/{field}")

print("\nOpening with ndv.imshow()...")
print("You should see sliders for:")
print("  - Position (p): 6 positions")
print("  - Time (t): 3 timepoints")
print("  - Channel (c): 1 channel")

ndv.imshow(z)
