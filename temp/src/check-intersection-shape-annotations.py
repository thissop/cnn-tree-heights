import geopandas as gpd

fs = ['data/standalone-fake/vector_annotation_1.gpkg',
      'data/test-dataset/vector_annotation_0.gpkg', 
      'data/test-dataset/vector_annotation_4.gpkg']

for f in fs: 
    gdf = gpd.read_file(f)
    print(gdf)