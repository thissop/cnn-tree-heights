#NOTE(Jesse):  The premise is to transform the CNN detection mask into polygons from which shadow lengths and heights
# are estimated from the associated mosaic cutline geometry's viewing and solar geometry.

UNTESTED

tile_shp_fp = "/path/to/mosaic/cutline.shp"
nn_result_fp = "/path/to/nn_result.tif"
out_dir = "/path/to/output/directory/"

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
    nn_mem_ds = gdal.GetDriverByName("MEM").CreateCopy(nn_dsk_ds)
    nn_mem_band = nn_mem_ds.GetRasterBand(1)

    arr = nn_mem_band.ReadAsArray()
    arr[arr == 255] = 0 #NOTE(Jesse): Mask out no data values
    arr[arr <= 50] = 0
    arr[arr > 50] = 1
    nn_mem_band.WriteArray(arr)
    arr = None

    nn_dsk_ds = None

    nn_mem_geo_ds = ogr.GetDriverByName("Memory").CreateDataSource("", None)
    nn_mem_geo_lyr = nn_mem_geo_ds.CreateLayer("shadows", gdal.GetSpatialRef(nn_mem_ds), ogr.wkbPolygon, None)

    new_field_names = ("shadow length", "height")
    shadow_length_idx = 0
    height_idx = 0
    for fn in new_field_names:
        nn_mem_geo_lyr.CreateField(ogr.FieldDefn(fn, ogr.OFTReal))

    gdal.Polygonize(nn_mem_band, nn_mem_band, nn_mem_geo_lyr, -1)

    nn_mem_band = None
    nn_mem_ds = None

    cutline_mem_ds = ogr.GetDriverByName("Memory").Open(tile_shp_fp)
    cutline_lyr = cutline_mem_ds.GetLayer(1)

    #NOTE(Jesse): debug
    c_lyr_defn = cutline_lyr.GetLayerDefn()
    for i in range(c_lyr_defn.GetFieldCount()):
        print(f"{c_lyr_defn.GetFieldDefn(i).GetName()}: {i}")

    off_nadir_idx = 0
    sun_elevation_idx = 1
    #DO ASSERTS

    set_feature = nn_mem_geo_lyr.SetFeature
    for c_ftr in cutline_lyr:
        c_geo = c_ftr.GetGeometryRef()
        if not c_geo.IsValid():
            #TODO(Jesse): Investigate how to robustly recover the geometry via MakeValid
            continue

        if c_geo.Area() < 300:
            continue #NOTE(Jesse): Arbiturary threshold to skip small cutline geometries

        off_nadir = c_ftr.GetFieldAsDouble(off_nadir_idx)
        sun_elevation = c_ftr.GetFieldAsDouble(sun_elevation_idx)

        nn_mem_geo_lyr.SetSpatialFilter(c_geo)
        for n_ftr in nn_mem_geo_lyr:
            #TODO(Jesse): Shadow length and height estimation
            shadow_length = 1
            height = 1
            n_ftr.SetField(shadow_length_idx, shadow_length)
            n_ftr.SetField(height_idx, height)
            set_feature(n_ftr)

    ds = ogr.GetDriverByName("GPKG").CreateCopy(join(out_dir, "heights.gpkg"), nn_mem_geo_ds)
    ds = None

main()
