import geopandas as gpd 

def count_polys(): 
    gdf = gpd.read_file('/Users/yaroslav/Documents/Work/NASA/data/first-annotations-push/cutout_2/annotations_2.gpkg')
    print(gdf)

    # 0: 309 
    # 1: 59
count_polys()