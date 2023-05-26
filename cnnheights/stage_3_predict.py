#NOTE(Jesse): The premise is to apply a trained CNN to detect tree canopies in the provided mosaic tile.

tile_fp = "/path/to/tile.tif"
out_fp = "/path/to/destination/"
model_weights_fp = "/path/to/weights.h5"


if __name__ != "__main__":
    print(f"This script {__name__} must be called directly and not imported to be used as a library.  Early exiting.")
    exit()

def main():
    from os.path import isfile, isdir, normpath, join
    from numpy import zeros, float32, ceil, uint8, maximum, nan, nanmean, nanstd, squeeze, isnan
    from osgeo import gdal

    gdal.UseExceptions()

    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "TRUE")
    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
    gdal.SetConfigOption("NUM_THREADS", "ALL_CPUS")
    gdal.SetCacheMax(0)

    global tile_fp
    global model_weights_fp
    global out_fp

    tile_fp = normpath(tile_fp)
    model_weights_fp = normpath(model_weights_fp)
    out_fp = normpath(out_fp)

    assert isfile(tile_fp), tile_fp
    assert isfile(model_weights_fp), model_weights_fp
    assert isdir(out_fp), out_fp

    from unet.UNet import UNet
    from unet.config import input_shape, batch_size

    #NOTE(Jesse): batch_size is how many 256x256 patches the CNN predicts in 1 go
    model = UNet([batch_size, *input_shape], 1, weight_file=model_weights_fp)

    tile_ds = gdal.Open(tile_fp)
    tile_x = tile_ds.RasterXSize
    tile_y = tile_ds.RasterYSize

    if tile_fp.endswith("mosaic.tif"):
        assert tile_x == 32768
        assert tile_x == tile_y

    geotransform = tile_ds.GetGeoTransform()
    out_projection = tile_ds.GetProjection()

    pan = tile_ds.GetRasterBand(1).ReadAsArray()
    ndvi = tile_ds.GetRasterBand(2).ReadAsArray()

    tile_ds = None

    overlap = 32
    step = 256 - overlap
    #NOTE(Jesse): batch_count is just how much data we pre-package to send to the CNN.  predict() will cut it into individual batches for us
    # Here, we cut the mosaic tile into rows which span the whole tile, and each such row is 1 batch
    batch_count = int(ceil(tile_x / step))
    batch = zeros((batch_count, 256, 256, 2), dtype=float32)
    out_predictions = zeros((tile_y, tile_x), dtype=uint8)

    def standardize(i):
        f_i = i.astype(float32)
        f_i[f_i == 0] = nan

        s_i = (f_i - nanmean(f_i)) / nanstd(f_i)
        s_i[isnan(s_i)] = 0

        if s_i.shape != (256, 256): #NOTE(Jesse): Occurs on the last xy step (a partial final step)
            s_i.resize((256, 256), refcheck=False)

        return s_i

    for y0 in range(0, tile_y, step):
        y1 = min(y0 + 256, tile_y)

        pan_batch = pan[y0:y1, :]
        ndvi_batch = ndvi[y0:y1, :]
        for batch_idx, x0 in enumerate(range(0, tile_x, step)):
            x1 = min(x0 + 256, tile_x)

            batch[batch_idx, ..., 0] =  standardize(pan_batch[:, x0:x1])
            batch[batch_idx, ..., 1] = standardize(ndvi_batch[:, x0:x1])

        predictions = squeeze(model.predict(batch, batch_size=16)) * 100
        batch.fill(0)

        out_predictions_batch = out_predictions[y0:y1, :]
        for i, x0 in enumerate(range(0, tile_x, step)):
            x1 = min(x0 + 256, tile_x)

            op = out_predictions_batch[:, x0:x1]
            p = predictions[i].astype(uint8)

            if op.shape != p.shape: #NOTE(Jesse): Handle fractional step patches
                p.resize(op.shape, refcheck=False)

            out_predictions_batch[:, x0:x1] = maximum(op, p)

    no_data_value = 255
    out_predictions[(ndvi == 0) & (out_predictions <= 50)] = no_data_value #NOTE(Jesse): Transfer no-data value fromo mosaic tile to these results.

    pan = None
    ndvi = None
    batch = None

    nn_mem_ds = gdal.GetDriverByName("MEM").Create('', xsize=tile_x, ysize=tile_y, bands=1, eType=gdal.GDT_Byte)
    assert nn_mem_ds

    nn_mem_ds.GetRasterBand(1).SetNoDataValue(no_data_value)
    nn_mem_ds.SetGeoTransform(geotransform)
    nn_mem_ds.SetProjection(out_projection)
    nn_mem_ds.GetRasterBand(1).WriteArray(out_predictions)

    out_predictions = None

    disk_create_options: list = ['COMPRESS=ZSTD', 'ZSTD_LEVEL=1', 'INTERLEAVE=BAND', 'Tiled=YES', 'NUM_THREADS=ALL_CPUS', 'SPARSE_OK=True', 'PREDICTOR=2']
    nn_disk_ds = gdal.GetDriverByName('GTiff').CreateCopy(join(out_fp, 'NN_classification.tif'), nn_mem_ds, options=disk_create_options)
    assert nn_disk_ds

    nn_mem_ds = None
    nn_disk_ds = None

main()
