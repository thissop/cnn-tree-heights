import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt 
from cnnheights.utilities import get_cutline_data
import matplotlib.pyplot as plt
from shapely.geometry import LineString

info = get_cutline_data(lat=1706523.96, lon=446623.30)
annotations_gdf = gpd.read_file('/Users/yaroslav/Documents/Work/NASA/layers/first-working-input/annotations_1.gpkg')
annotations_gdf.set_crs('epsg:4326', allow_override=True)


centroids = annotations_gdf.centroid

d = 3
dy = np.abs(d/np.tan(np.radians(info['SUN_AZ'])))
print(dy)
lines = [LineString([(x-d, x+d), (y+dy, y-dy)]) for x, y in zip(centroids.x, centroids.y)]
lines_gdf = gpd.GeoDataFrame({'geometry':lines},geometry='geometry', crs=annotations_gdf.crs)

print(lines_gdf.iloc[0]['geometry'])

fig, ax = plt.subplots()

annotations_gdf.plot(ax=ax, color='#408ee0')

for i in range(len(lines_gdf.index)):
    xy = lines_gdf['geometry'][i].xy
    x = [xy[0][0], xy[1][0]]
    y = [xy[0][1], xy[1][1]]
    ax.plot(x, y, color='black', linewidth=0.1)

centroids.plot(ax=ax, color='indianred', markersize=0.2, zorder=3)

# lines_gdf.plot(ax=ax, color='black')

ax.set(xlabel='Longitude', ylabel='Latitude')

plt.savefig('/Users/yaroslav/Documents/GitHub/cnn-tree-heights/src/monthly/jan2023/library-testing/first-layer/shadow_lengths_demo_approach.png', dpi=250)