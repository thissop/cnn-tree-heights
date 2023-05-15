import geopandas as gpd 
import matplotlib.pyplot as plt
import numpy as np

lines_gdf = gpd.read_file('temp_lines_gdf_debug_invalid.gpkg')
annotations_gdf = gpd.read_file('temp_annotations_gdf_debug_invalid.gpkg')

flags = np.array([i.is_valid for i in annotations_gdf['geometry']])
mask = np.where(flags==True)[0] 
mask_ = np.where(flags!=True)[0]


fig, axs = plt.subplots(1, 2) 

annotations_gdf['geometry'][mask].plot(ax=axs[0]) 
lines_gdf['geometry'][mask].plot(ax=axs[0], lw=0.5, color='black') 

annotations_gdf['geometry'][mask_].plot(ax=axs[1]) 
lines_gdf['geometry'][mask_].plot(ax=axs[1], lw=0.5, color='black') 

axs[0].set(title='Valid')
axs[1].set(title='Invalid')

plt.savefig('diagnostics/valid-invalid-10e(lines).pdf')