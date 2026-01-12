# /// script
# dependencies = [
#     "ndv[pyqt,vispy,ngff]",
# ]
# ///
from __future__ import annotations

import ndv

SOURCE = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5"

# If you have a file saved with the NGFF Zarr specification, you can simply pass
# the path or URL to `ndv.imshow` to visualize it.

data = f"{SOURCE}/idr0062A/6001240_labels.zarr"  # single position
# data = f"{SOURCE}/idr0033A/BR00109990_C2.zarr"  # multi position
# data = f"{SOURCE}/idr0090/190129.zarr"  # plate

ndv.imshow(data)
