import rasterio 
background_tif = '/Users/yaroslav/Documents/Work/NASA/data/jesse/big mosaic/big mosaic.tif'
src = rasterio.open(background_tif)

print(src.height)
print(src.width)