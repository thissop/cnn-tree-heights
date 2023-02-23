import geopandas as gpd 
import rasterio 
import numpy as np
from rasterio import mask 
predictions = gpd.read_file('/Users/yaroslav/Documents/GitHub/cnn-tree-heights/src/monthly/feb2023/saving-predictions/predicted_polygons.shp')
geom = predictions.iloc[4]['geometry'] # [bbox.geometry]`

r'''
cutout_polys = [f'/Users/yaroslav/Documents/Work/NASA/data/old/july2022-testing-input/thaddaeus_vector_rectangle_{i}.gpkg' for i in range(1,11)]
for i in cutout_polys: 
    print(i.split('/')[-1])
    i = gpd.read_file(i)
    print(i.contains(geom))
    #print(i.bounds)
'''

img = rasterio.open('/Users/yaroslav/Documents/Work/NASA/data/old/july2022-testing-input/ndvi_thaddaeus_training_area_1.tif')

fake_arr = np.zeros(img.shape)
print(img.shape)
#print(fake_arr) # [statement], huh 
#print(img.shape)
#arr = img.read()
#print(arr[0][234])

out_image, out_transform=mask.mask(img, [geom], crop=True)
out_image = out_image[0]
print(out_image)
print(out_image.shape)
print(out_image.mask)