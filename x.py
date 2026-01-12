from rich import print

from yaozarrs import open_group, validate_ome_object
from yaozarrs import open_group, validate_ome_object, validate_zarr_store

import ndv

# path = "/Users/fdrgsp/Desktop/acqz_hcs_zarr_example.ome.zarr"
path = "/Users/fdrgsp/Desktop/ts_hcs_zarr_example.ome.zarr"
# path = "/Users/fdrgsp/Desktop/zarr_example.ome.zarr"
# make sure the zarr store is valid
# print(path)
# validate_zarr_store(path)

# open with yaozarrs
z = open_group(path)
print(type(z))
validate_ome_object(z.ome_metadata())
print(z.ome_metadata())
ndv.imshow(z)

# open with zarr
# z1 = zarr.open_group(path, mode="r")
# print('INFO', z1.info)
# print(type(z1))
# ndv.imshow(z1)

# open from path
# ndv.imshow(path)

# import tifffile
# p = "/Users/fdrgsp/Desktop/hcs_multi_ome_tiff/hcs_p000.ome.tiff"
# t = tifffile.TiffFile(p)
# ndv.imshow(t.asarray())
