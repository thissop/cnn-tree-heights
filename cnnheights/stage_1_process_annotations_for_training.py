#NOTE(Jesse): The premise is to generate binary raster masks of the geometry annotations.  We also provide another mask
# that segments the space between nearby geometries (they are scaled and any overlap is considered "nearby"), which is used during training to aid in closed canopy disaggregation.

# Instead of providing 2 separate files or bands of binary masks, they are interwoven into the same stream.  Values of 1 corrospond to geometry pixels
# and values of 2 corrospond to overlap pixels.  0 means "neither geometry nor overlap" and is marked as "no data" in the GDAL band.

# This script intends to injest data from the stage_0 script.

training_data_fp = "/path/to/training/annotations"

def PROCESS_compute_tree_annotation_and_boundary_raster(vector_fp):
    #NOTE(Jesse): Find an accompanying raster filename based on the vector_fp to geolocate the result masks.
    vector_fp_split = vector_fp.split('/')
    vector_fn = vector_fp_split[-1]
    raster_fp_base = "/".join(vector_fp_split[:-1]) + "/"
    vector_fn = vector_fn.split('.')[0]
    v_id = vector_fn.split('_')[-1]
    assert v_id.isdigit(), f"{vector_fn} the ID number was expected after the last '_' in the file name."

    #TODO(Jesse): Remove PNG support and use the pan_ndvi_cutout file name convention

    raster_disk_ds = gdal.Open(raster_fp_base + f"extracted_ndvi_{int(v_id)}.png")

    #NOTE(Jesse): Create in memory raster of the same geospatial extents as the mask for high performance access.
    raster_mem_ds = gdal.GetDriverByName("MEM").Create('', xsize=raster_disk_ds.RasterXSize, ysize=raster_disk_ds.RasterYSize, bands=1, eType=gdal.GDT_Byte)
    band = raster_mem_ds.GetRasterBand(1)
    band.SetNoDataValue(255)
    raster_mem_ds.SetGeoTransform(raster_disk_ds.GetGeoTransform())
    raster_mem_ds.SetProjection(raster_disk_ds.GetProjection())
    band.Fill(0)
    del raster_disk_ds

    #NOTE(Jesse): Similarly with the vector polygons.  Load from disk and into a memory dataset.
    vector_disk_ds = gdal.OpenEx(vector_fp, gdal.OF_VECTOR)
    vector_mem_ds = gdal.GetDriverByName("Memory").Create('', 0, 0, 0, gdal.GDT_Unknown) #NOTE(Jesse): GDAL has a highly unintuitive API
    vector_mem_ds.CopyLayer(vector_disk_ds.GetLayer(0), 'orig')
    del vector_disk_ds

    #NOTE(Jesse): 'Buffer' extends the geometry out by the geospatial unit amount, approximating 'scaling' by 1.5.
    #             OGR, believe it or not, does not have an easy way to scale geometries like this.
    #             SQL is our only performant recourse to apply these operations to the data within OGR.
    sql_layer = vector_mem_ds.ExecuteSQL("select Buffer(GEOMETRY, 1.5, 5) from orig", dialect="SQLITE")
    vector_mem_ds.CopyLayer(sql_layer, 'scaled') #NOTE(Jesse): The returned 'layer' is not part of the original dataset for some reason? Requires a manual copy.
    del sql_layer

    #NOTE(Jesse): "Burn" the unscaled vector polygons into the raster image.
    opt_orig = gdal.RasterizeOptions(bands=[1], burnValues=1, layers='orig')
    gdal.Rasterize(raster_mem_ds, vector_mem_ds, options=opt_orig)

    #NOTE(Jesse): Track which pixels were burned into (via the '1') here, and reuse the band later.
    orig_arr = band.ReadAsArray()
    orig_arr_mask = orig_arr == 1
    band.Fill(0)

    #NOTE(Jesse): Burn the scaled geometries with the 'add' option, which will add the burn value to the destination pixel
    #             for all geometries which overlap it.  Basically, create a heatmap.
    opt_scaled = gdal.RasterizeOptions(bands=[1], burnValues=1, layers='scaled', add=True)
    gdal.Rasterize(raster_mem_ds, vector_mem_ds, options=opt_scaled)

    #NOTE(Jesse): Retain pixels with burn values > 1 (0 means no polygon overlap, 1 means 1 polygon overlaps, and >2 means multiple overlaps)
    composite_arr = band.ReadAsArray()
    composite_arr[composite_arr > 1] = 2 #NOTE(Jesse): 2 means overlap
    composite_arr[composite_arr == 1] = 0 #NOTE(Jesse): 0 means no polygon coverage
    composite_arr[orig_arr_mask] = 1 #NOTE(Jesse): 1 means original canopy

    #NOTE(Jesse): Save the composite array out to disk.
    disk_create_options = ['COMPRESS=ZSTD', 'ZSTD_LEVEL=1', 'INTERLEAVE=BAND', 'Tiled=YES', 'NUM_THREADS=ALL_CPUS', 'SPARSE_OK=True', 'PREDICTOR=2', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256']
    result_fp = raster_fp_base + f"annotation_and_boundary_{v_id}.tif"
    raster_disk_ds = gdal.GetDriverByName("GTiff").Create(result_fp, xsize=raster_mem_ds.RasterXSize, ysize=raster_mem_ds.RasterYSize, bands=1, eType=gdal.GDT_Byte, options=disk_create_options)
    raster_disk_ds.GetRasterBand(1).SetNoDataValue(0) #NOTE(Jesse): 0 is actually "valid" but for QGIS visualization, we want it to be transparent.
    raster_disk_ds.SetGeoTransform(raster_mem_ds.GetGeoTransform())
    raster_disk_ds.SetProjection(raster_mem_ds.GetProjection())
    raster_disk_ds.GetRasterBand(1).WriteArray(composite_arr)
    del raster_disk_ds

    return result_fp

from osgeo import gdal
gdal.UseExceptions()

gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "TRUE")
gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
gdal.SetConfigOption("NUM_THREADS", "ALL_CPUS")

#NOTE(Jesse): The functions and imports above have to be global and not barred as below to be available for spawned processes.
# But, we don't want them to import MP Pool nor run main at all.  This is less about restricting shared code use and more about
# kosher MP semantics.

if __name__ == "__main__":
    from multiprocessing import Pool

    def main():
        from time import time
        start = time()

        from os import listdir
        from os.path import join, normpath

        global training_data_fp

        training_data_fp = normpath(training_data_fp)

        training_files = listdir(training_data_fp)
        training_files = [join(training_data_fp, f) for f in training_files if f.endswith(".gpkg")]

        assert len(training_files) > 0, f"No training .gpkg database files were found in {training_data_fp}"

        with Pool() as p:
            fps = p.map(PROCESS_compute_tree_annotation_and_boundary_raster, training_files, chunksize=1)

        vrt_dsses = [gdal.Open(fp) for fp in len(fps)]
        vrt_ds = gdal.BuildVRT(join(training_data_fp, "annotation_and_boundary.vrt"), vrt_dsses)
        vrt_ds = None

        vrt_dsses = None

        end = time()
        elapsed_minutes = (end - start) / 60.0
        print(elapsed_minutes)

    main()
