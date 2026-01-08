"""Example: Generate and view NGFF files."""

import tempfile
from pathlib import Path

import ndv

# Create temporary directory
temp_dir = Path(tempfile.mkdtemp())
print(f"Creating NGFF files in: {temp_dir}\n")

print("Creating multi-position OME-Zarr file...")
mp_path = ndv.data.ngff_multi_position(
    str(temp_dir / "multipos.ome.zarr"),
    n_positions=5,
    shape=(3, 2, 64, 64),  # T=3, C=2, Y=64, X=64
    dtype="uint8",
)
print(f"✅ Created: {mp_path}")

ndv.imshow(mp_path)
