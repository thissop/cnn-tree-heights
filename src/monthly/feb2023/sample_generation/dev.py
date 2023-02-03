import rasterio 

background_tif = '/Users/yaroslav/Documents/Work/NASA/data/jesse/big mosaic/big mosaic.tif'

# randomly choose left corner 
# eventually, make sure no overlap with other regions 

import matplotlib.pyplot as plt
from rasterio import plot
from rasterio.windows import transform
import rasterio

fig, ax = plt.subplots()

dataset = rasterio.open(background_tif)
window = rasterio.windows.Window(0, 0, 1056, 1056)
img = dataset.read(1, window=window)
window_transform = transform(window, dataset.transform)
extent = plot.plotting_extent(img, window_transform)

# load img here using rasterio
plot.show(img, origin='upper', transform=window_transform, extent=extent, interpolation=None, ax=ax)
ax.set_aspect('equal')
plt.show()


'''
with rasterio.open('tests/data/RGB.byte.tif') as src:
    r, g, b = src.read()
'''