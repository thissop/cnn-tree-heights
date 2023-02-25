# Annotation and Extraction

## Summary Notes
- Band 1 is PAN (red)
- Band 2 is NDVI (green)

## Preparing Workspace 
- Open the big image file. Duplicate the layer. For reference, red is PAN, green is NDVI. 
- Export the current view you want to work in to it's own GeoTIFF (ending will be .tif; export by current layer extent so you only get one layer in each). 
- Close the big map, open the small one. 
- Double click on original layer, go to symbology --> select singleband grey for render type
- Change grey band to Band 2 (NDVI)
- Perform similiar changes to copied layer, but set a min of 250 and max of 750 work well. Keep band to band 1 (default)
- Return to symbology for first layer. Enable the Colorize box, choose the Lighten Blending mode.
- export to ndvi and pan png images. do this by going to raster --> clip raster by extent (set bounds to calculate from the cutout tiff image layer)
- from advanced paramters, set additional command line parameters `-b 1` for pan and `-b 2` for ndvi
- save the coordinates of the center of the image for use in my intersection function later

## Annotations 
- Make new geopackage layer (needs to be in its own folder). Needs to be a .gpkg file. 
- Set geometry type to polygon
- make sure it's wgs 84
- select that layer you just made and hit pencil
- to add polygon annotations, cmd+.
- to save polygon anotation, right click enter (to go back to pan, shift+alt+p)
- to save layer and annotation, click yellow pencil again (and save changes)
- to export geopackage annotation, go to raster-->extraction-->clip raster by extent 
- set input layer, set clipping extent (by selection of image, or full image because I kinda like that better), then set the clip output file (make it a .tif, same folder as the gpkg layer). Run extraction. 
