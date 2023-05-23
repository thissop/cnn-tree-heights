#NOTE(Jese): The premise is to generate N (10) cutouts of size 1024x1024 pixels in parallel
# and then reject overlapping ones.
#
# The process repeats until at least N non-overlapping cutouts are produced, which is the expected first result.
#
# While in general there is not a scientific reason why overlapped regions are problematic,
# instead, it wastes the annotator's time annotating regions they've already annotated.
#
#NOTE: For now, the directory where cutouts are to be stored will be checked to be empty and will only proceed with an empty directory.
#TODO: In the future, this script may be extended to confirm non-overlapping directories and/or regenerate directories with overlapping cutouts.

if __name__ != "__main__":
    print(f"This script {__name__} must be called directly and not imported to be used as a library.  Early exiting.")
    exit()

#NOTE(Jesse): These are the two inputs you must configure, and both must already exist.  We do not create directories on your behalf.
mosaic_fp = "/path/to/mosaic.tif" #NOTE(Jesse): This is the filepath to the mosaic image from which cutouts are selected.
cutout_fp = "/path/to/cutouts" #NOTE(Jesse): Directory to store selected cutouts.
bounds_to_generate = 2

def is_overlapping_1d(range1, range2):
    return range1[0] <= range2[1] & range1[1] >= range2[0]

def filter_overlapped_bounds(bounds_xy):
    assert len(bounds_xy) > 0

    #TODO(Jesse): Bug where some overlapped tiles are not filtered

    bounds_xy = list(bounds_xy)

    filtered_count: int = 0
    #NOTE(Jesse): Iterate backwards for inline removal of list elements
    for i in range(len(bounds_xy) - 1, -1, -1):
        top_bound_xy = bounds_xy[i]
        box1 = (top_bound_xy[0], top_bound_xy[0] + 1024, #NOTE(Jesse): x_min, x_max
                top_bound_xy[1], top_bound_xy[1] + 1024) #NOTE(Jesse): y_min, y_max

        for bottom_bound_xy in bounds_xy:
            if top_bound_xy is bottom_bound_xy:
                continue

            box2 = (bottom_bound_xy[0], bottom_bound_xy[0] + 1024,
                    bottom_bound_xy[1], bottom_bound_xy[1] + 1024)

            #NOTE(Jesse) 1D check in both X and Y to determine overlap
            if is_overlapping_1d(box1[:2], box2[:2]) & is_overlapping_1d(box1[2:], box2[2:]):
                del bounds_xy[i]
                filtered_count += 1

                break

    print(f"Filtered {filtered_count} bounds")

    return bounds_xy


def main():
    from os import listdir
    from os.path import normpath, join, isdir, isfile

    import rasterio

    from osgeo import gdal
    gdal.UseExceptions()

    from numpy.random import default_rng

    global cutout_fp
    global mosaic_fp
    global bounds_to_generate

    cutout_fp = normpath(cutout_fp)
    mosaic_fp = normpath(mosaic_fp)

    assert isfile(mosaic_fp), mosaic_fp
    assert isdir(cutout_fp), cutout_fp

    cutout_files = listdir(cutout_fp)
    cutout_files = [c for c in cutout_files if not c.startswith('.')]
    if len(cutout_files) > 0:
        print("Cutout files already exist in cutout directory.  Must be empty.  Early exiting.")
        return

    rng = default_rng()
    randint = rng.integers

    mosaic_xy = 32768 - 1 #NOTE(Jesse): Mosaic sizes are standardized
    cutout_xy = 1024

    extent_xy = mosaic_xy - cutout_xy
    while True:
        bounds_xy = randint(0, extent_xy, (bounds_to_generate * 2, 2))
        bounds_xy = filter_overlapped_bounds(bounds_xy)
        if len(bounds_xy) >= bounds_to_generate:
            break

    bounds_xy = bounds_xy[:bounds_to_generate]

    vrt_dsses = [None] * len(bounds_xy)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN=True, GDAL_NUM_THREADS="ALL_CPUS", NUM_THREADS="ALL_CPUS"):
        with rasterio.open(mosaic_fp, 'r') as ds: #TODO(Jesse): Multiprocessing if this is slow somehow.
            ds_transform = ds.transform
            Window = rasterio.windows.Window
            transform = rasterio.windows.transform

            profile = ds.profile
            profile['width'] = cutout_xy
            profile['height'] = cutout_xy
            profile['interleave'] = 'band'
            profile['predictor'] = 2

            for i, bound_xy in enumerate(bounds_xy):
                output_tif_path = join(cutout_fp, f"pan_ndvi_cutout_{i}.tif")

                window = Window(bound_xy[0], bound_xy[1], cutout_xy, cutout_xy)
                profile['transform'] = transform(window, ds_transform)

                with rasterio.open(output_tif_path, 'w', **profile) as new_data_set:
                    new_data_set.write(ds.read(window=window))

                vrt_dsses[i] = gdal.Open(output_tif_path)

    vrt_ds = gdal.BuildVRT(join(cutout_fp, "cutout.vrt"), vrt_dsses)
    vrt_ds = None
    vrt_dsses = None

from time import time
begin = time() / 60

main()

end = time() / 60
elapsed_minutes = end - begin
print(f"Took {elapsed_minutes} minutes")
