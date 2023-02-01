import geopandas as gpd 

annotations_gpkg = '/Users/yaroslav/Documents/Work/NASA/layers/first-working-input/annotations_1.gpkg'

annotations_gdf = gpd.read_file(annotations_gpkg)

annotations_gdf = annotations_gdf.set_crs('epsg:32637', allow_override=True)
print(list(annotations_gdf))