project_base_dir: str = '/projects/sciteam/bbbo/data/'
project_base_dir: str = '/Users/jrmeyer3/Desktop/NASA/trees/'

from os.path import isdir
if not isdir(project_base_dir):
    print(f'[ERROR] Project base directory {project_base_dir} not found. Exiting.')
    exit()

from cython.cy_determine_tree_heights import determine_tree_heights_internal, merge_results

global_mosaic_pan_data: object = None
def read_pan(mosaic_sub_tile: str):
    import gdal
    gdal.UseExceptions()

    global global_mosaic_pan_data

    mosaic_ds = gdal.Open(mosaic_sub_tile)
    if mosaic_ds is None:
        print('[ERROR] Could not open {}'.format(mosaic_sub_tile))
        return

    global_mosaic_pan_data = mosaic_ds.GetRasterBand(1).ReadAsArray()

    mosaic_ds = None

def extract_zip(temp_dir: str, result_zip: str):
    from zipfile import ZipFile
    from os import rename
    from os.path import isfile

    with ZipFile(result_zip) as archive:
        files_in_archive: list = archive.namelist()
        if 'features.gpkg' not in files_in_archive:
            print('[ERROR] features.gpkg not found in zip archive. Exiting. ')
            return

        _ = archive.extract('features.gpkg', temp_dir)
        rename(temp_dir + '/features.gpkg', temp_dir + '/tmp_features.gpkg') #NOTE(Jesse): Have not found a way to do this at extraction stage.
        assert isfile(temp_dir + '/tmp_features.gpkg')


def determine_tree_heights(result_zip: str):#shp_mem_ds, mosaic_pan_data, cutline_file: str): #NOTE(Jesse): Local variables are faster than global variables, so I lump everything into a function habitually now.
    global global_mosaic_pan_data

    from time import time

    before_total_minutes: c.double = time() / 60

    from gc import collect

    from os.path import isfile, isdir
    from tempfile import TemporaryDirectory
    from os import system, listdir, cpu_count
    from sys import stderr
    from threading import Thread

    import ogr
    ogr.UseExceptions()

    split_result_raster: list = result_zip.split('/')
    mosaic_dir_intermediate: list = split_result_raster[-1].split('_')

    '''
    from:
        /projects/sciteam/bazc/data/NeuralNet/Results/Run_1/SSA_32628_GE01-QB02-WV02-WV03-WV04_001_004_mosaic_1_1.zip

    to:
        /projects/sciteam/bazc/data/SSA/32628/GE01-QB02-WV02-WV03-WV04/mosaic/001_004/1_1/SSA_32628_GE01-QB02-WV02-WV03-WV04_PAN_NDVI_001_004_mosaic_1_1.zip
    '''

    mosaic_dir: str = project_base_dir + mosaic_dir_intermediate[0] + '/' + mosaic_dir_intermediate[1] + '/' + mosaic_dir_intermediate[2] + '/' + mosaic_dir_intermediate[5] + '/' + mosaic_dir_intermediate[3] + '_' + mosaic_dir_intermediate[4] + '/' + mosaic_dir_intermediate[6] + '_' + mosaic_dir_intermediate[7][:-4] + '/'
    assert isdir(mosaic_dir), f'Mosaic dir {mosaic_dir} not found.'

    cutline_dir: str = project_base_dir + mosaic_dir_intermediate[0] + '/' + mosaic_dir_intermediate[1] + '/' + mosaic_dir_intermediate[2] + '/' + mosaic_dir_intermediate[5] + '/' + mosaic_dir_intermediate[3] + '_' + mosaic_dir_intermediate[4] + '/'
    assert isdir(cutline_dir), f'Cutline dir {cutline_dir} not found.'

    #SSA_32628_GE01-QB02-WV02-WV03-WV04_PAN_NDVI_001_002_mosaic_cutlines.shp
    last_instrument: str = 'WV04' #NOTE(Jesse): We only support rev2 now
    mosaic_tile_intermediate: list = split_result_raster[-1].split(last_instrument)
    cutline_file: str = cutline_dir + mosaic_tile_intermediate[0] + last_instrument + '_PAN_NDVI' + mosaic_tile_intermediate[1][:-8] + '_cutlines.shp' #"SSA_32628_GE01-QB02-WV02-WV03-WV04_PAN_NDVI_010_007_mosaic_cutlines.shp" #"SSA_32628_GE01-QB02-WV02-WV03-WV04_PAN_NDVI_009_006_mosaic_cutlines.shp"
    assert isfile(cutline_file), f'{cutline_file} not found.'

    mosaic_sub_tile: str = mosaic_dir + mosaic_tile_intermediate[0] + last_instrument + '_PAN_NDVI' + mosaic_tile_intermediate[1][:-4] + '.tif' #"SSA_32628_GE01-QB02-WV02-WV03-WV04_PAN_NDVI_009_006_mosaic_3_4.tif"
    if not isfile(mosaic_sub_tile):
        print("[ERROR] Could not locate {}. Exiting.".format(mosaic_sub_tile), file=stderr)
        return

    print('Reading in PAN band.')
    read_pan_thread = Thread(target=read_pan, args=(mosaic_sub_tile,))
    read_pan_thread.start()

    with TemporaryDirectory() as temp_dir:
        vector_file: str = temp_dir + '/tmp_features.gpkg'

        print('Unzipping {}'.format(result_zip))
        extract_thread = Thread(target=extract_zip, args=(temp_dir, result_zip))
        extract_thread.start()

        extract_thread.join()
        read_pan_thread.join()

        vector_disk_ds = ogr.Open(vector_file, 1)
        if vector_disk_ds is None:
            print(f'{vector_file} could not be opened! Exiting.', file=stderr)
            return

        tree_disk_layer = vector_disk_ds.GetLayerByName('trees')
        tree_layer_defn = tree_disk_layer.GetLayerDefn()
        has_rsme = tree_layer_defn.GetFieldIndex('rsme') != -1
        has_shadow_length = tree_layer_defn.GetFieldIndex('shadow_length') != -1
        tree_layer_defn = None

        #NOTE(Jesse): Add these attributes now since we didn't realize we needed them when the databases were first formed.
        if not has_rsme:
            tree_disk_layer.CreateField(ogr.FieldDefn("rsme", ogr.OFTReal))

        if not has_shadow_length:
            #NOTE(Jesse): OK, so if we have spaces in the field name, we have to query for it using 'field name' in SQL, which will CONVERT THE STORAGE TYPE TO STRING
            #            It's totally insane, so to prevent that (and I had incredible joy debugging this (:  ), we just use '_' inplace of a space.
            tree_disk_layer.CreateField(ogr.FieldDefn("shadow_length", ogr.OFTReal))
        tree_disk_layer = None

        vector_mem_ds = ogr.GetDriverByName('Memory').CopyDataSource(vector_disk_ds, '')
        tree_mem_layer = vector_mem_ds.GetLayerByName('trees')

        max_tree_fid: c.int = vector_disk_ds.ExecuteSQL('SELECT MAX(fid) FROM trees').GetFeature(0).GetField(0)
        feature_count: c.int = tree_mem_layer.GetFeatureCount()

        assert (max_tree_fid + 1) == feature_count #NOTE(Jesse): If this passes for all runs we can remove the SELECT statement.
        print(f'Determining heights of at most {max_tree_fid + 1} trees.')

        vector_disk_ds = None

        #NOTE(Jesse): The sample size for the model fit are driven by the following parameters.
        #TODO(Jesse): tune
        confidence_interval_z: c.float = 2.576 #NOTE(Jesse): 99%. 95% is for suckers.
        std_from_mean: c.float = 0.5
        margin_of_error: c.float = .05

        #NOTE(Jesse): https://en.wikipedia.org/wiki/Sample_size_determination
        required_sample_size: c.long = int(((confidence_interval_z * confidence_interval_z) * (std_from_mean - (std_from_mean * std_from_mean))) / (margin_of_error * margin_of_error)) * 4 #NOTE(Jesse): We pad this out a bit since we filter out many (40-60%) canidate features

        if max_tree_fid < required_sample_size: #TODO(Jesse): We actually need this per cutline, which we do check for later.
            print(f'Unsufficient features. Require at least {required_sample_size} samples but found had {max_tree_fid}.')
            return

        cutline_ds = ogr.Open(cutline_file)
        if cutline_ds is None:
            print('Could not open {cutline_file}, exiting.', file=stderr)
            return

        if cutline_ds.GetLayerCount() != 1:
            print(f"[ERROR] Cutline {cutline_file} dataset should only have 1 layer. Exiting.", file=stderr)
            return

        cutline_layer = cutline_ds.GetLayer(0)
        cutline_geometry_count: c.int = cutline_layer.GetFeatureCount()

        print('Generating per cutline job payloads.')
        trees_per_thread: int = 65_536

        cutline_threshold_payloads: list = []
        job_index: c.int = 0
        for i in range(cutline_geometry_count):
            cutline = cutline_layer.GetFeature(i)
            cutline_geo = cutline.GetGeometryRef()

            cutline_geo_count: c.int = cutline_geo.GetGeometryCount()
            largest_cutline_polygon_index: c.int = 0

            cutline_geo_area: c.float = 0.0
            for j in range(cutline_geo_count):
                cutline_geo_polygon = cutline_geo.GetGeometryRef(j)
                assert cutline_geo_polygon.GetGeometryName() == 'POLYGON'

                this_cutline_geo_polygon_area: c.float = cutline_geo_polygon.Area()
                if this_cutline_geo_polygon_area > cutline_geo_area:
                    largest_cutline_polygon_index = j
                    cutline_geo_area = this_cutline_geo_polygon_area

                cutline_geo_polygon = None
                cutline_geo = cutline_geo.GetGeometryRef(largest_cutline_polygon_index)

            if cutline_geo.Area() < 50_000.0: #NOTE(Jesse): Skip small cutlines.
                continue

            tree_mem_layer.SetSpatialFilter(cutline_geo)
            trees_count: c.int = tree_mem_layer.GetFeatureCount()

            if trees_count > 0 and trees_count >= required_sample_size:
                tree_layer_extent: tuple = tree_mem_layer.GetExtent() #NOTE(Jesse): Divide space evenly amongst all threads. This means some features may be processed twice.
                long_min = tree_layer_extent[0]
                long_max = tree_layer_extent[1]

                lat_min = tree_layer_extent[2]
                lat_max = tree_layer_extent[3]

                lat_diff: float = lat_max - lat_min

                num_threads: int = int(max(1, (trees_count / trees_per_thread) + 0.5)) #NOTE(Jesse): Round area up.
                lat_per_thread: float = lat_diff / num_threads

                for j in range(num_threads):
                    cutline_threshold_payloads.append((job_index, i, cutline_file, vector_file, required_sample_size, (long_min, long_max, lat_min + (lat_per_thread * j), lat_min + (lat_per_thread * (j + 1)))))
                    job_index += 1

        tree_mem_layer.SetSpatialFilter(None)
        cutline = None
        cutline_layer = None
        cutline_ds = None

        print('Done')

        if len(cutline_threshold_payloads) == 0:
            print('No features intersect with {cutline_file}, exiting.', file=stderr)
            return

        #NOTE(Jesse): Below is a 'serial' version of below, for testing!
        #for payload in cutline_threshold_payloads:
        #    tree_heights_dispatch(payload)

        import multiprocessing as mp
        fork_ctx = mp.get_context("fork")
        assert fork_ctx

        before_map_minutes: c.double = time() / 60

        thread_count: int = cpu_count()# * 2 #NOTE(Jesse): cpu_count() does not factor in CPU hyperthreads, so double the number for our HW.
        with fork_ctx.Pool(thread_count) as p:
            p.map(tree_heights_dispatch, cutline_threshold_payloads, chunksize=1)
        cutline_threshold_payloads_count: int = len(cutline_threshold_payloads)
        cutline_threshold_payloads = None

        global_mosaic_pan_data = None
        collect()

        print('Done')

        after_map_minutes: c.double = time() / 60
        print(f'Took {after_map_minutes - before_map_minutes} minutes to compute tree heights.')

        tmp_files: list = [temp_dir + '/' + tmp_file for tmp_file in listdir(temp_dir) if 'split_' in tmp_file]
        assert len(tmp_files) <= cutline_threshold_payloads_count

        if len(tmp_files) == 0:
            print('[WARN] No features in all cutlines were processable. Exiting.', file=stderr)
            return

        print('Merging results.')
        before_merge_minutes: c.double = time() / 60

        merge_results(tree_mem_layer, tmp_files) #NOTE(Jesse): tree_mem_layer is mutated in-place.
        tree_mem_layer = None
        tmp_files = None

        after_merge_minutes: c.double = time() / 60
        print(f'Took {after_merge_minutes - before_merge_minutes} minutes to merge results.')

        print('Write outputs to disk')
        dst_vector_disk_ds = ogr.GetDriverByName('GPKG').CopyDataSource(vector_mem_ds, vector_file.replace('tmp_features', 'features'))
        if dst_vector_disk_ds is None:
            print('[ERROR] Memory dataset could not be saved to disk. Exiting.')
            return

        vector_mem_ds = None
        dst_vector_disk_ds = None
        print('Done')

        print(f'Saving and Archiving results.')
        #system('7z a {} {}\\features.gpkg -y'.format(zip_file, temp_dir)) # NOTE (Eric): This is the syntax for 7-zip on Windows.
        system(f'zip -jf {result_zip} {temp_dir}/features.gpkg')

        after_total_minutes: c.double = time() / 60
        print(f'Took a total of {after_total_minutes - before_total_minutes} minutes to complete.')


def tree_heights_dispatch(args):
    return determine_tree_heights_internal(args, global_mosaic_pan_data)

if __name__ == "__main__":
    from sys import argv

    try:
        determine_tree_heights(argv[1])
    except Exception as e:
        print(e)
