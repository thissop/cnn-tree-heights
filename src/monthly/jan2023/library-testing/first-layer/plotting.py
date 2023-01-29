import matplotlib.pyplot as plt
import geopandas as gpd



gdf = gpd.read_file('/Users/yaroslav/Documents/Work/NASA/layers/first-working-input/annotations_1.gpkg')
gdf.set_crs('epsg:3857', allow_override=True)
print(gdf)
fig, ax = plt.subplots()

gdf.plot(ax=ax)

centroids = gdf.centroid

centroids.plot(ax=ax, color='brown')

plt.show()