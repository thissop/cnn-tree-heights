import geopandas as gpd
import numpy as np
import pandas as pd

def first_try():

    cutlines_gdf = gpd.read_file('/Users/yaroslav/Documents/Work/NASA/data/jesse/thaddaeus_cutline/SSAr2_32628_GE01-QB02-WV02-WV03-WV04_PAN_NDVI_010_003_mosaic_cutlines.shp')

    cutlines_gdf = cutlines_gdf.set_crs('EPSG:32637', allow_override=True)

    annotations_gdf = gpd.read_file('/Users/yaroslav/Documents/Work/NASA/layers/first-working-input/annotations_1.gpkg')

    annotations_gdf = annotations_gdf.set_crs('EPSG:32637', allow_override=True)

    contains = cutlines_gdf.contains(annotations_gdf) # returns all false? 

    print(cutlines_gdf)
    for i in range(len(cutlines_gdf.index)): 
        print(cutlines_gdf.iloc[i]['geometry'].contains(annotations_gdf.iloc[0]['geometry'])) # returns all false?


