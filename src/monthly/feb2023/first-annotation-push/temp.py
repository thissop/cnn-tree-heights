import geopandas as gpd 
import time 
from cnnheights.utilities import shadows_from_annotations, height_from_shadow
from cnnheights.plotting import plot_shadow_lengths
def count_polys():

    polys = []

    for i in [0,1,2,3]:  
        gdf = gpd.read_file(f'/Users/yaroslav/Documents/Work/NASA/data/first-annotations-push/cutout_{i}/annotations_{i}.gpkg')
        polys.append(len(gdf.index))

    # 0: 309 
    # 1: 66
    # 2: 165

    print(polys)
    print(sum(polys))

count_polys()

quit()

def make_shadow_gdfs(): 

    # kinda takes forever to do these ngl...

    s1 = time.time()
    af0 = '/Users/yaroslav/Documents/Work/NASA/data/first-annotations-push/cutout_0/annotations_0.gpkg'
    af1 = '/Users/yaroslav/Documents/Work/NASA/data/first-annotations-push/cutout_1/annotations_1.gpkg'
    af2 = '/Users/yaroslav/Documents/Work/NASA/data/first-annotations-push/cutout_2/annotations_2.gpkg'

    background_0 = '/Users/yaroslav/Documents/Work/NASA/data/first-annotations-push/cutout_0/cutout_0.tif'
    cutlines = '/Users/yaroslav/Documents/Work/NASA/data/jesse/thaddaeus_cutline/SSAr2_32628_GE01-QB02-WV02-WV03-WV04_PAN_NDVI_010_003_mosaic_cutlines.shp'
    shadows_gdf_path_0 = 'src/monthly/feb2023/first-annotation-push/shadows_gdf_0.gpkg'
    shadows_gdf_0 = shadows_from_annotations(af0, cutlines_shp=cutlines, north=1691780.62, east=464709.85, epsg='32628', save_path=shadows_gdf_path_0)
    shadows_0_plot = 'src/monthly/feb2023/first-annotation-push/plots/annotations/annotations_0.pdf'
    plot_shadow_lengths(shadows_gdf_0, background_tif=background_0, save_path=shadows_0_plot)

    print('t1', time.time()-s1)
    
    s2 = time.time()
    background_2 = '/Users/yaroslav/Documents/Work/NASA/data/first-annotations-push/cutout_2/cutout_2.tif'
    shadows_gdf_path_2 = 'src/monthly/feb2023/first-annotation-push/shadows_gdf_2.gpkg'
    shadows_gdf_2 = shadows_from_annotations(af2, cutlines_shp=cutlines, north=1706616.040, east=447832.579, epsg='32628', save_path=shadows_gdf_path_2)
    shadows_2_plot = 'src/monthly/feb2023/first-annotation-push/plots/annotations/annotations_2.pdf'
    plot_shadow_lengths(shadows_gdf_2, background_tif=background_2, save_path=shadows_2_plot)
    print('t2', time.time()-s2)

#make_shadow_gdfs()

def longest_line(ring=None): 
    from shapely.geometry import LinearRing
    import numpy as np

    coords = list([list(i) for i in LinearRing([(-1, 0), (1, 3), (4, -2)]).coords])
    coords = coords+[coords[0]]
    diameters = []
    for i in range(len(coords)-1): 
        left = coords[i]
        right = coords[i+1]
        dist = np.sqrt((left[0]-right[0])**2+(left[1]-right[1])**2)
        diameters.append(dist)
    
    diameter = np.max(diameters)

    return diameter

#longest_line()

def plot_diagnostics(): 
    from cnnheights.plotting import plot_annotation_diagnostics
    import time 

    af0 = '/Users/yaroslav/Documents/Work/NASA/data/first-annotations-push/cutout_0/annotations_0.gpkg'
    af1 = '/Users/yaroslav/Documents/Work/NASA/data/first-annotations-push/cutout_1/annotations_1.gpkg'
    af2 = '/Users/yaroslav/Documents/Work/NASA/data/first-annotations-push/cutout_2/annotations_2.gpkg'


    s1 = time.time()
    background_0 = '/Users/yaroslav/Documents/Work/NASA/data/first-annotations-push/cutout_0/cutout_0.tif'
    cutlines = '/Users/yaroslav/Documents/Work/NASA/data/jesse/thaddaeus_cutline/SSAr2_32628_GE01-QB02-WV02-WV03-WV04_PAN_NDVI_010_003_mosaic_cutlines.shp'
    shadows_gdf_0 = shadows_from_annotations(af0, cutlines_shp=cutlines, north=1691780.62, east=464709.85, epsg='32628')
    shadows_0_plot = '/Users/yaroslav/Documents/GitHub/cnn-tree-heights/src/monthly/feb2023/first-annotation-push/plots/annotations/annotations_0.pdf'
    plot_shadow_lengths(shadows_gdf_0, background_tif=background_0, save_path=shadows_0_plot)

    print('t1', time.time()-s1)
    
    s1 = time.time()
    background_1 = '/Users/yaroslav/Documents/Work/NASA/data/first-annotations-push/cutout_1/cutout_1.tif'
    shadows_gdf_1 = shadows_from_annotations(af1, cutlines_shp=cutlines, north=1693717.05, east=449094.71, epsg='32628')
    shadows_1_plot = '/Users/yaroslav/Documents/GitHub/cnn-tree-heights/src/monthly/feb2023/first-annotation-push/plots/annotations/annotations_1.pdf'
    plot_shadow_lengths(shadows_gdf_1, background_tif=background_1, save_path=shadows_1_plot)
    print('t1', time.time()-s1)

    s2 = time.time()
    background_2 = '/Users/yaroslav/Documents/Work/NASA/data/first-annotations-push/cutout_2/cutout_2.tif'
    shadows_gdf_2 = shadows_from_annotations(af2, cutlines_shp=cutlines, north=1706616.040, east=447832.579, epsg='32628')
    shadows_2_plot = '/Users/yaroslav/Documents/GitHub/cnn-tree-heights/src/monthly/feb2023/first-annotation-push/plots/annotations/annotations_2.pdf'
    plot_shadow_lengths(shadows_gdf_2, background_tif=background_2, save_path=shadows_2_plot)
    print('t2', time.time()-s2)

    plot_path_0 = 'src/monthly/feb2023/first-annotation-push/plots/diagnostics/diagnostic_0.pdf'
    plot_annotation_diagnostics(shadows_gdf_0, plot_path_0)

    plot_path_1 = 'src/monthly/feb2023/first-annotation-push/plots/diagnostics/diagnostic_1.pdf'
    plot_annotation_diagnostics(shadows_gdf_1, plot_path_1)

    plot_path_2 = 'src/monthly/feb2023/first-annotation-push/plots/diagnostics/diagnostic_2.pdf'
    plot_annotation_diagnostics(shadows_gdf_2, plot_path_2)

plot_diagnostics()