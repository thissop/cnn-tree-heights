#NOTE(Jese): The premise is to generate N non-overlapping regions of 1024x1024 pixels in parallel for annotation or prediction purposes
#
# While this can be used locally, it is intended to be used on NCCS Discover.


if __name__ != "__main__":
    print(f"This script {__name__} must be called directly and not imported to be used as a library.  Early exiting.")
    exit()

#NOTE(Jesse): These are the two inputs you must configure, and both must already exist.  We do not create directories on your behalf.
tile_root_dir = "/css/nga/mosaics/SSAc3.2/UTM_Zones/" 
cutout_dir = "/discover/nobackup/jrmeyer3/tree_heights/annotation_cutouts/" #NOTE(Jesse): Directory to store selected cutouts.
chirps_fp = "/discover/nobackup/projects/sacs_tucker/inputs/chirps/mean_chirps_05deg/mean_annual_chirps.tif"
bounds_to_generate_per_tile = 3

local = False
if local:
    tile_root_dir = "/Users/jrmeyer3/Desktop/NASA/UTM_Zones"
    cutout_dir = "/Users/jrmeyer3/Desktop/NASA/tree height/cutouts"
    chirps_fp = "/Users/jrmeyer3/Desktop/NASA/mean_annual_chirps_int16_zstd.tif"

def is_overlapping_1d(range1, range2):
    return (range1[0] <= range2[1]) & (range1[1] >= range2[0])

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
    from os import listdir, rename
    from os.path import normpath, join, isdir, isfile

    from gc import collect

    import rasterio
    from rasterio.windows import Window, from_bounds, transform
    from rasterio.crs import CRS

    from osgeo import gdal
    gdal.UseExceptions()

    from numpy.random import default_rng
    from numpy import squeeze
    from random import shuffle

    #NOTE(Jesse): Chirps rainfall imports
    from shapely.ops import transform as transform_ops
    from shapely import box
    
    from pyproj import Transformer, Proj

    global cutout_dir
    global tile_root_dir
    global bounds_to_generate_per_tile
    global chirps_fp

    cutout_dir = normpath(cutout_dir)
    tile_root_dir = normpath(tile_root_dir)
    chirps_fp = normpath(chirps_fp)

    assert isdir(tile_root_dir), tile_root_dir
    assert isdir(cutout_dir), cutout_dir
    assert isfile(chirps_fp), chirps_fp

    cutout_files = listdir(cutout_dir)
    cutout_files = [c for c in cutout_files if not c.startswith('.')]
    if len(cutout_files) > 0:
        print("Cutout files already exist in cutout directory.  Must be empty.  Early exiting.")
        return

    #TODO(Jesse): Gather all previously generated cutouts and filter newly generated ones from them. low priority

    rng = default_rng()
    randint = rng.integers

    mosaic_xy = 32768 - 1 #NOTE(Jesse): Mosaic sizes are standardized
    cutout_xy = 1024

    extent_xy = mosaic_xy - cutout_xy
    attempts_max = 10

    utm_zones = (28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40)
    tiles_per_zone = 1
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN=True, GDAL_NUM_THREADS="ALL_CPUS", NUM_THREADS="ALL_CPUS"):
        with rasterio.open(chirps_fp) as chirps_ds:
            chirps_crs = chirps_ds.crs
            chirps_trans = chirps_ds.transform
            assert "EPSG:4326" == chirps_crs.to_string()

            cutout_count = 0
            for utm in utm_zones: #TODO(Jesse): multiprocess map
                collect()

                utm_tile_dir = join(tile_root_dir, "PV", f"326{utm}")
                assert isdir(utm_tile_dir), utm_tile_dir

                utm_vrt_dsses = []
                utm_crs = CRS.from_epsg(32600 + utm)

                #NOTE(Jesse): Build a UTM -> WGS84 transformer so we can use the cutout spatial bounds to read the associated CHIRPS values.
                utm_to_wgs84_trans = Transformer.from_proj( #NOTE(Jesse): Transform image bound's crs to training area's crs for intersection tests.
                    Proj(utm_crs), #NOTE(Jesse): SRC
                    Proj(chirps_crs), #NOTE(Jesse): DST
                    always_xy = True #NOTE(Jesse): lat / long swapped in proj2
                ).transform

                #TODO(Jesse): At some point we may want to bias towards higher rainfall regions, as that's where the trees are.
                utm_tile_dirs = listdir(utm_tile_dir)
                utm_tile_dirs = [d for d in utm_tile_dirs if len(d) == 7 and d[:3].isalnum() and d[4:].isalnum()]
                shuffle(utm_tile_dirs)
                randomly_selected_tile_numbers = utm_tile_dirs[:tiles_per_zone]
                for tile_number in randomly_selected_tile_numbers:
                    bounds_xy = None
                    for _ in range(attempts_max):
                        bounds_xy = filter_overlapped_bounds(randint(0, extent_xy, (bounds_to_generate_per_tile * 2, 2))) #NOTE(Jesse): Generate more than requested in the event of overlaps.
                        if len(bounds_xy) >= bounds_to_generate_per_tile:
                            bounds_xy = bounds_xy[:bounds_to_generate_per_tile]
                            break
                    else:
                        print(f"[NOTE] Could not generate requested bounds count {bounds_to_generate_per_tile} in the number of attempts {attempts_max}. {bounds_to_generate_per_tile - len(bounds_xy)} not produced")
                        print(f"\tThis is likely because the requested count was too large, and the odds for overlap given the extent size {extent_xy} was too high.")
                    
                    mosaic_name = f"SSAr2_326{utm}_GE01-QB02-WV02-WV03_PV_{tile_number}_mosaic.tif"
                    mosaic_fp = join(utm_tile_dir, tile_number, mosaic_name)
                    if not isfile(mosaic_fp):
                        mosaic_name = f"SSAr2_326{utm}_GE01-QB02-WV02-WV03-WV04_PV_{tile_number}_mosaic.tif"
                        mosaic_fp = join(utm_tile_dir, tile_number, mosaic_name)
                        assert isfile(mosaic_fp), mosaic_fp

                    with rasterio.open(mosaic_fp, 'r') as ds:
                        ds_transform = ds.transform

                        profile = ds.profile
                        profile['width'] = cutout_xy
                        profile['height'] = cutout_xy
                        profile['interleave'] = 'band'
                        profile['predictor'] = 2

                        for bound_xy in bounds_xy:
                            tmp_output_tif_path = join(cutout_dir, f"utm_{utm}_pan_ndvi_cutout_{tile_number}_{cutout_count}_tmp.tif")
                            cutout_count += 1

                            window = Window(bound_xy[0], bound_xy[1], cutout_xy, cutout_xy)
                            profile['transform'] = transform(window, ds_transform)
                            with rasterio.open(tmp_output_tif_path, 'w', **profile) as new_data_set:
                                new_data_set.write(ds.read(window=window))

                                cutout_bounds_to_wgs84 = transform_ops(utm_to_wgs84_trans, box(*new_data_set.bounds)).bounds
                                w = from_bounds(*cutout_bounds_to_wgs84, chirps_trans)
                                w = Window(int(w.col_off + (w.width / 2)), int(w.row_off + (w.height / 2)), 1, 1) #NOTE(Jesse): Center point sample
                                chirps = squeeze(chirps_ds.read(1, window=w))
                                new_data_set.update_tags(CHIRPS = f"{chirps}")

                            output_tif_path = tmp_output_tif_path.replace("_tmp", "")
                            rename(tmp_output_tif_path, output_tif_path)
                            utm_vrt_dsses.append(gdal.Open(output_tif_path))

                    vrt_ds = gdal.BuildVRT(join(cutout_dir, f"utm_{utm}_cutout.vrt"), utm_vrt_dsses)
                    vrt_ds = None
                    utm_vrt_dsses = None

from time import time
begin = time()

failure = False
try:
    main()
except Exception as e:
    print(e)
    failure = True

end = time()
elapsed_minutes = (end - begin) / 60
print(f"Took {elapsed_minutes} minutes")

if failure:
    print("[FAILURE]")
