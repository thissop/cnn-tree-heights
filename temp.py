import matplotlib.pyplot as plt 
import geopandas as gpd 

gdf = gpd.read_file('/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/temp/predictions/predicted_polygons.shp')
fig, ax = plt.subplots()
gdf.plot(ax=ax)
plt.savefig('/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/temp/temp.png')