import geopandas as gpd 
import matplotlib.pyplot as plt
import numpy as np

lines_gdf = gpd.read_file('temp_lines_gdf_debug_invalid.gpkg')
print(lines_gdf.length)