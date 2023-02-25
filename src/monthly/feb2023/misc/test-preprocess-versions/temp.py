import geopandas as gpd 
import numpy as np 

annotations = []  
for f in [f'/Users/yaroslav/Documents/Work/NASA/old/data/first-annotations-push/individual_cutouts/cutout_{i}/annotations_{i}.gpkg' for i in (0,1,3,4)]:  
    gdf = gpd.read_file(f) 
    annotations.append(len(gdf['geometry']))  
 
print(annotations) # [309, 137, 370, 535]
print(np.mean(annotations)) # 337.75