import geopandas as gpd

predictions_file = '/Users/yaroslav/Documents/GitHub/cnn-tree-heights/src/monthly/feb2023/saving-predictions/predicted_polygons.shp'
crs = 'EPSG:32628'

gdf = gpd.read_file(predictions_file)

new_geometries = []

for i in gdf['geometry']:
    if i.type == 'Polygon': 
        new_geometries.append(i)

d = {'geometry':new_geometries}

new_gdf = gpd.GeoDataFrame(d, crs=crs)
new_gdf.to_file(predictions_file)
