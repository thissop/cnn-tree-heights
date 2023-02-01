from cnnheights.utilities import shadows_from_annotations

annotations_gpkg = '/Users/yaroslav/Documents/Work/NASA/layers/first-working-input/annotations_1.gpkg'
cutlines_shp = '/Users/yaroslav/Documents/Work/NASA/data/jesse/thaddaeus_cutline/SSAr2_32628_GE01-QB02-WV02-WV03-WV04_PAN_NDVI_010_003_mosaic_cutlines.shp'
save_path = '/Users/yaroslav/Documents/GitHub/cnn-tree-heights/src/monthly/jan2023/standardizing-extant-work/shadows_gdf.feather'

print(shadows_from_annotations(annotations_gpkg, cutlines_shp, north=1706575.98, east=446542.55, epsg='32628'))
