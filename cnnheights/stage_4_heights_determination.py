#NOTE(Jesse):  The premise is to transform the CNN detection mask into polygons from which shadow lengths and heights
# are estimated from the associated mosaic cutline geometry's viewing and solar geometry.

tile_shp_fp = "/path/to/mosaic_tile.shp"
nn_result_fp = "/path/to/nn_results.tif"
out_dir = "/path/to/output/"

if __name__ != "__main__":
    print(f"This script {__name__} must be called directly and not imported to be used as a library.  Early exiting.")
    exit()

def main():
    from os.path import normpath, isfile, isdir, join

    global tile_shp_fp
    global nn_result_fp
    global out_dir

    tile_shp_fp = normpath(tile_shp_fp)
    nn_result_fp = normpath(nn_result_fp)
    out_dir = normpath(out_dir)

    assert isfile(tile_shp_fp), tile_shp_fp
    assert isfile(nn_result_fp), nn_result_fp
    assert isdir(out_dir), out_dir

    from osgeo import ogr, gdal
    gdal.UseExceptions()
    ogr.UseExceptions()

    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "TRUE")
    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
    gdal.SetConfigOption("NUM_THREADS", "ALL_CPUS")
    gdal.SetCacheMax(0)

    nn_dsk_ds = gdal.Open(nn_result_fp)
    nn_mem_ds = gdal.GetDriverByName("MEM").CreateCopy("", nn_dsk_ds)
    nn_mem_band = nn_mem_ds.GetRasterBand(1)

    arr = nn_mem_band.ReadAsArray()
    arr[arr == 255] = 0 #NOTE(Jesse): Mask out no data values
    arr[arr <= 50] = 0
    arr[arr > 50] = 1
    nn_mem_band.WriteArray(arr)
    arr = None

    nn_dsk_ds = None

    nn_mem_geo_ds = ogr.GetDriverByName("Memory").CreateDataSource("")
    nn_mem_geo_lyr = nn_mem_geo_ds.CreateLayer("shadows", nn_mem_ds.GetSpatialRef(), ogr.wkbPolygon)

    new_field_names = ("shadow length", "height")
    shadow_length_idx = 0
    height_idx = 1
    for fn in new_field_names:
        nn_mem_geo_lyr.CreateField(ogr.FieldDefn(fn, ogr.OFTReal))

    gdal.Polygonize(nn_mem_band, nn_mem_band, nn_mem_geo_lyr, -1)

    nn_mem_band = None
    nn_mem_ds = None

    cutline_dsk_ds = ogr.Open(tile_shp_fp)
    cutline_mem_ds = ogr.GetDriverByName("Memory").CopyDataSource(cutline_dsk_ds, "") #NOTE(Jesse): Yes, OGR and GDAL swap param order here.
    cutline_dsk_ds = None
    cutline_lyr = cutline_mem_ds.GetLayer(0)

    debug = False
    if debug:
        c_lyr_defn = cutline_lyr.GetLayerDefn()
        for i in range(c_lyr_defn.GetFieldCount()):
            print(f"{c_lyr_defn.GetFieldDefn(i).GetName()}: {i}")

    #NOTE(Jesse): Field indices, via the above debug code
    cat = 0
    IMAGENAME = 1
    SENSOR = 2
    ACQDATE = 3
    CAT_ID = 4
    RESOLUTION = 5
    OFF_NADIR = 6
    SUN_ELEV = 7
    SUN_AZ = 8
    SAT_ELEV = 9
    SAT_AZ = 10
    CLOUDCOVER = 11
    TDI = 12
    DATE_DIFF = 13
    SCORE = 14
    SCAN_DIR = 15
    RF_MULT = 16
    RF_ADD = 17
    STATS_MIN = 18
    STATS_MAX = 19
    STATS_STD = 20
    STATS_MEAN = 21
    STATS_PXCT = 22

    set_feature = nn_mem_geo_lyr.SetFeature
    for i, c_ftr in enumerate(cutline_lyr):
        c_geo = c_ftr.GetGeometryRef()
        if not c_geo.IsValid():
            #TODO(Jesse): Investigate how to robustly recover the geometry via MakeValid
            continue

        if c_geo.Area() < 300:
            continue #NOTE(Jesse): Arbiturary threshold to skip small cutline geometries

        off_nadir = c_ftr.GetFieldAsDouble(OFF_NADIR)
        sun_elevation = c_ftr.GetFieldAsDouble(SUN_ELEV)

        nn_mem_geo_lyr.SetSpatialFilter(c_geo)
        for n_ftr in nn_mem_geo_lyr:
            #TODO(Jesse): Shadow length and height estimation
            shadow_length = i
            height = i
            n_ftr.SetField(shadow_length_idx, shadow_length)
            n_ftr.SetField(height_idx, height)
            set_feature(n_ftr)

    nn_mem_geo_lyr.SetSpatialFilter(None)
    nn_mem_geo_lyr = None
    ds = ogr.GetDriverByName("GPKG").CopyDataSource(nn_mem_geo_ds, join(out_dir, "heights.gpkg"))
    ds = None

from time import time
start = time()
main()
stop = time()
print(f"Took {(stop - start) / 60} minutes")
