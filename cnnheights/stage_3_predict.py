#NOTE(Jesse): The premise is to apply a trained CNN to detect tree canopies in the provided mosaic tile.
# The outputs are a raster and vector dataset of the predicted shadow geometries.

tile_fp = "/path/to/tile.tif"
out_fp = "/path/to/destination/"
model_weights_fp = "/path/to/weights.h5"

label_name = "shadows" #NOTE(Jesse): What is being predicted?
vector_simplification_amount = 0.45 #NOTE(Jesse): Are predicted vectory geometries to be simplified?  Set to > 0 to enable, in accordance with OGR Geometry Simplify.

if __name__ != "__main__":
    print(f"This script {__name__} must be called directly and not imported to be used as a library.  Early exiting.")
    exit()

def main():
    print("Import")
    from time import time
    from os import rename, environ, remove
    from os.path import isfile, isdir, normpath, join
    from numpy import zeros, float32, uint8, maximum, nan, nanmean, nanstd, squeeze, mean, std
    from osgeo import gdal, ogr
    import sozipfile.sozipfile as zipfile

    from gc import collect

    gdal.UseExceptions()
    ogr.UseExceptions()

    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "TRUE")
    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
    gdal.SetConfigOption("NUM_THREADS", "ALL_CPUS")
    gdal.SetCacheMax(0)

    global tile_fp
    global model_weights_fp
    global out_fp
    global label_name
    global vector_simplification_amount

    assert label_name is not None
    assert vector_simplification_amount is not None
    assert isinstance(vector_simplification_amount, (int, float))

    tile_fp = normpath(tile_fp)
    model_weights_fp = normpath(model_weights_fp)
    out_fp = normpath(out_fp)

    assert isfile(tile_fp), tile_fp
    assert isfile(model_weights_fp), model_weights_fp
    assert isdir(out_fp), out_fp

    environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    environ["TF_GPU_THREAD_MODE"] = "gpu_private" #NOTE(Jesse): Seperate I/O and Compute CPU thread scheduling.

    #environ["TF_XLA_FLAGS"] = "--xla_gpu_persistent_cache_dir=C:/Users/jrmeyer3/Desktop/NASA/trees/:"

    print("UNET")
    from unet.UNet import UNet
    from unet.config import input_shape, batch_size

    assert input_shape[0] == input_shape[1]
    shape_xy = input_shape[0]

    model = UNet([batch_size, *input_shape], 1, weight_file=model_weights_fp)
    model.compile(jit_compile=True)

    tile_ds = gdal.Open(tile_fp)
    tile_x = tile_ds.RasterXSize
    tile_y = tile_ds.RasterYSize

    if tile_fp.endswith("mosaic.tif"):
        assert tile_x == 32768
        assert tile_x == tile_y

    tile_fn = tile_fp.split('/')[-1].split('.')[0]

    geotransform = tile_ds.GetGeoTransform()
    out_projection = tile_ds.GetProjection()

    print("Read tile bands")
    pan = tile_ds.GetRasterBand(1).ReadAsArray()
    ndvi = tile_ds.GetRasterBand(2).ReadAsArray()

    tile_ds = None

    step_xy = int(shape_xy * 0.875) #NOTE(Jesse): ~12% overlap between patches fed to the UNet.
    batch = zeros((batch_size, *input_shape), dtype=float32)
    out_predictions = zeros((tile_y, tile_x), dtype=uint8)

    def standardize(i):
        nonlocal shape_xy, float32

        f_i = i.astype(float32) if i.dtype != float32 else i
        has_nans = f_i == 0
        if has_nans.any():
            f_i[has_nans] = nan

            s_i = (f_i - nanmean(f_i)) / nanstd(f_i)
            s_i[has_nans] = 0
        else:
            s_i = (f_i - mean(f_i)) / std(f_i)

        if s_i.shape != (shape_xy, shape_xy): #NOTE(Jesse): Occurs on the last xy step (a partial final step)
            s_i.resize((shape_xy, shape_xy), refcheck=False)

        return s_i

    print("Predict")
    batch_predict = model.predict_on_batch

    y0_out = x0_out = batch_idx = 0
    y1_out = x1_out = shape_xy
    for y0 in range(0, tile_y, step_xy):
        y1 = min(y0 + shape_xy, tile_y)

        for x0 in range(0, tile_x, step_xy):
            x1 = min(x0 + shape_xy, tile_x)

            batch[batch_idx, ..., 0] =  standardize(pan[y0:y1, x0:x1])
            batch[batch_idx, ..., 1] = standardize(ndvi[y0:y1, x0:x1])

            batch_idx += 1
            if batch_idx == batch_size:
                batch_idx = 0

                predictions = squeeze(batch_predict(batch)) * 100
                batch.fill(0)

                for p in predictions:
                    p = p.astype(uint8)
                    op = out_predictions[y0_out:y1_out, x0_out:x1_out]

                    if p.shape != op.shape: #NOTE(Jesse): Handle fractional step patches
                        p.resize(op.shape, refcheck=False)

                    out_predictions[y0_out:y1_out, x0_out:x1_out] = maximum(op, p)
                    p = None

                    x0_out += step_xy
                    x1_out = min(x0_out + shape_xy, tile_x)
                    assert x0_out != x1_out
                    if x0_out >= tile_x:
                        x0_out = 0
                        x1_out = shape_xy

                        y0_out += step_xy
                        y1_out = min(y0_out + shape_xy, tile_y)
                        assert y1_out != y0_out

    no_data_value = 255
    out_predictions[(ndvi == 0) & (out_predictions <= 50)] = no_data_value #NOTE(Jesse): Transfer no-data value from mosaic tile to these results.

    model = None
    predictions = None
    pan = None
    ndvi = None
    batch = None

    collect()

    nn_mem_ds = gdal.GetDriverByName("MEM").Create('', xsize=tile_x, ysize=tile_y, bands=1, eType=gdal.GDT_Byte)
    assert nn_mem_ds

    nn_mem_ds.GetRasterBand(1).SetNoDataValue(no_data_value)
    nn_mem_ds.SetGeoTransform(geotransform)
    nn_mem_ds.SetProjection(out_projection)
    nn_mem_band = nn_mem_ds.GetRasterBand(1)

    print("Creating prediction mask")
    #NOTE(Jesse): Convert raster predictions to vector geometries via binary mask of >50% thresholding
    arr = out_predictions.copy()

    arr[arr == no_data_value] = 0
    arr[arr <= 50] = 0
    arr[arr > 50] = 1
    nn_mem_band.WriteArray(arr)
    arr = None

    ogr_mem_ds = ogr.GetDriverByName("Memory").CreateDataSource("")
    ogr_mem_lyr = ogr_mem_ds.CreateLayer(label_name, nn_mem_ds.GetSpatialRef(), ogr.wkbPolygon)

    print("Polygonize")
    gdal.Polygonize(nn_mem_band, nn_mem_band, ogr_mem_lyr, -1)

    print("Simplify")
    if vector_simplification_amount > 0:
        #ogr_mem_lyr.StartTransaction()
        set_feature = ogr_mem_lyr.SetFeature
        for i, ftr in enumerate(ogr_mem_lyr):
            geo = ftr.GetGeometryRef()
            s_geo = geo.SimplifyPreserveTopology(vector_simplification_amount)
            assert s_geo.IsValid(), i

            ftr.SetGeometry(s_geo)

            assert ftr.Validate(), i

            set_feature(ftr)
            s_geo = None
            geo = None
        ftr = None
        #ogr_mem_lyr.CommitTransaction()

    print("Save out")
    out_tmp_fn = f"{label_name}_tmp.gpkg"
    ogr_dsk_ds = ogr.GetDriverByName("GPKG").CopyDataSource(ogr_mem_ds, join(out_fp, out_tmp_fn))
    ogr_mem_lyr = None
    ogr_mem_ds = None
    ogr_dsk_ds = None

    disk_create_options: list = ['COMPRESS=ZSTD', 'ZSTD_LEVEL=1', 'INTERLEAVE=BAND', 'Tiled=YES', 'NUM_THREADS=ALL_CPUS', 'SPARSE_OK=True', 'PREDICTOR=2']
    nn_mem_band.WriteArray(out_predictions)
    out_predictions = None

    nn_disk_ds = gdal.GetDriverByName('GTiff').CreateCopy(join(out_fp, 'NN_classification_tmp.tif'), nn_mem_ds, options=disk_create_options)
    nn_mem_ds = None
    nn_disk_ds = None

    zip_tmp_fn = join(out_fp, f'{tile_fn}_{label_name}_tmp.zip')
    with zipfile.ZipFile(zip_tmp_fn, 'w',
                        compression=zipfile.ZIP_DEFLATED,
                        chunk_size=zipfile.SOZIP_DEFAULT_CHUNK_SIZE) as myzip:

        myzip.write(join(out_fp, out_tmp_fn), arcname=out_tmp_fn.replace("_tmp", ""))
        myzip.write(join(out_fp, "NN_classification_tmp.tif"), arcname="NN_classification.tif")

    rename(zip_tmp_fn, zip_tmp_fn.replace("_tmp", ""))
    remove(join(out_fp, out_tmp_fn))
    remove(join(out_fp, 'NN_classification_tmp.tif'))

main()
