#!/usr/bin/env python3
"""
Test NDV with HCS (High Content Screening) Zarr files.

This demonstrates that ndv properly exposes HCS structure with:
- Position slider for all wells/fields
- Time, channel, and spatial dimensions from each field
"""

import ndv
import yaozarrs

print("=" * 80)
print("NDV HCS Support Test")
print("=" * 80)

# Test 1: HCS Plate with 3 wells x 2 fields = 6 positions
print("\n1. Opening HCS Plate (96-well)")
print("-" * 80)
hcs_path = "/Users/fdrgsp/Desktop/acqz_hcs_zarr_example.ome.zarr/96-well"
z_plate = yaozarrs.open_group(hcs_path)

meta = z_plate.ome_metadata()
print(f"Metadata type: {type(meta).__name__}")

if hasattr(meta, 'plate'):
    print(f"Plate name: {meta.plate.name}")
    print(f"Wells: {len(meta.plate.wells)}")

wrapper = ndv.models.DataWrapper.create(z_plate)
print(f"Dimensions: {wrapper.dims}")
print(f"Sizes: {wrapper.sizes()}")

print("\nPosition mapping:")
for i, (well, field) in enumerate(wrapper._position_info):
    print(f"  p={i}: {well}/{field}")

print("\n✓ Expected: Position slider with 6 positions")
print("✓ Expected: Time slider with 3 timepoints")
print("✓ Expected: Channel slider with 1 channel")

# Test data access
print("\nTesting data access...")
data_p0_t0_c0 = wrapper.isel({0: 0, 1: 0, 2: 0})  # p=0, t=0, c=0
print(f"  Position 0, Time 0, Channel 0: shape={data_p0_t0_c0.shape}, dtype={data_p0_t0_c0.dtype}")

data_p5_t2_c0 = wrapper.isel({0: 5, 1: 2, 2: 0})  # p=5, t=2, c=0
print(f"  Position 5, Time 2, Channel 0: shape={data_p5_t2_c0.shape}, dtype={data_p5_t2_c0.dtype}")

print("\n" + "=" * 80)
print("Opening with ndv.imshow()...")
print("=" * 80)
print("You should see:")
print("  - Position (p) slider: 0-5 (6 positions)")
print("  - Time (t) slider: 0-2 (3 timepoints)")
print("  - Channel (c) slider: 0 (1 channel)")
print("=" * 80)

ndv.imshow(z_plate)
