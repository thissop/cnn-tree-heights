import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt 
from cnnheights.utilities import get_cutline_data
import matplotlib.pyplot as plt
from shapely.geometry import LineString, LinearRing
import shapely 

# THIS IS TAKING HOURS TO GET INTERSECTION OF THESE GENERATED LINES WITH THE POLYs LOL

info = get_cutline_data(lat=1706523.96, lon=446623.30)
annotations_gdf = gpd.read_file('/Users/yaroslav/Documents/Work/NASA/layers/first-working-input/annotations_1.gpkg')
annotations_gdf.set_crs('epsg:4326', allow_override=True)

centroids = annotations_gdf.centroid

d = 3
dy = np.abs(d/np.tan(np.radians(info['SUN_AZ'])))
lines = [LineString([(x-d, y+dy), (x+d, y-dy)]) for x, y in zip(centroids.x, centroids.y)]
lines_gdf = gpd.GeoDataFrame({'geometry':lines}, geometry='geometry', crs=annotations_gdf.crs)

fig, ax = plt.subplots()

annotations_gdf.plot(ax=ax, color='#408ee0')
lines_gdf.plot(ax=ax, color='black', linewidth=0.1)
centroids.plot(ax=ax, color='indianred', markersize=0.2, zorder=3)

lengths = lines_gdf.intersection(annotations_gdf, align=False).length

for i in range(len(lengths)): 
    p = centroids[i]
    ax.annotate(f'{round(lengths[i], 2)}', xy=(1.000001*p.x, 1.000001*p.y))

print(lengths)

# lines_gdf.plot(ax=ax, color='black')
'''
import matplotlib.image as mpimg

img = mpimg.imread('/Users/yaroslav/Documents/Work/NASA/layers/first-working-input/ndvi_1.png')
plt.imshow(img)
plt.show()

ax.set(xlabel='Longitude', ylabel='Latitude')
'''
plt.savefig('/Users/yaroslav/Documents/GitHub/cnn-tree-heights/src/monthly/jan2023/library-testing/first-layer/shadow_lengths_demo_approach.png', dpi=450)
plt.savefig('/Users/yaroslav/Documents/GitHub/cnn-tree-heights/src/monthly/jan2023/library-testing/first-layer/shadow_lengths_demo_approach.svg')
plt.savefig('/Users/yaroslav/Documents/GitHub/cnn-tree-heights/src/monthly/jan2023/library-testing/first-layer/shadow_lengths_demo_approach.pdf')