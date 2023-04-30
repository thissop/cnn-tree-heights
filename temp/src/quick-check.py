import osgeo
from osgeo import gdal
fp = '/Users/yaroslav/Documents/Work/NASA/current/data/samples/mosaic-0-samples-0/unprocessed/cutout_0/raw_ndvi_0.tif'

d = gdal.Open(fp)
print(d.GetDescription())

# raster layer name is just the file name without the file extension? 
