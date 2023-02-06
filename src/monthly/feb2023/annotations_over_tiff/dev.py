from cnnheights.plotting import plot_shadow_lengths
from cnnheights.utilities import shadows_from_annotations

annotations_gpkg = '/Users/yaroslav/Documents/Work/NASA/layers/first-working-input/annotations_1.gpkg'
cutlines_shp = '/Users/yaroslav/Documents/Work/NASA/data/jesse/thaddaeus_cutline/SSAr2_32628_GE01-QB02-WV02-WV03-WV04_PAN_NDVI_010_003_mosaic_cutlines.shp'
background_tif = '/Users/yaroslav/Documents/Work/NASA/layers/first-working-input/cutout_1.tif'
save_path = '/Users/yaroslav/Documents/GitHub/cnn-tree-heights/src/monthly/feb2023/annotations_over_tiff/first_plot.pdf'

shadows_gdf = shadows_from_annotations(annotations_gpkg=annotations_gpkg, 
                                       cutlines_shp=cutlines_shp, 
                                       north=1706575.98, east=446542.55, epsg='32628')

plot_shadow_lengths(shadows_gdf, background_tif, save_path=save_path)