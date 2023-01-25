# cython: language_level=3, infer_types=True, cdivision=True, boundscheck=True, initializedcheck=True, nonecheck=True
# distutils: language=c

# Principle Author Eric Romero, improvements by Jesse Meyer
# NASA Goddard Space Flight Center
# January 28th 2020
# Fixed and debugged for BW testing on April 15th 2020

# Shadow Length and Tree Height Determination via Sun-Satellite Geometry

#ASSUMPTIONS:
# There is a single tree in the pan read window.
# There is a single shadow in pan read window which bec.longs to the given tree.
# The tree centroid is actually the tree centroid.
# There is a single shadow that is traversed when caluclating its length.
# The tree foliage is not highly specular in the pan imagery.
# The curve fit provides a good cut off value that indicates the value at which a given pan value is a shadow.

### IMPORTANT NOTE ####
# If running a local test on windows, be sure to replace all '/' with '\\' and vice versa for Mac OS/Linux
# If running local test on Windows, uncomment system call on line 681 and comment out line 682 (Again a Windows vs Mac OS/Linux syntax)


import cython as c #NOTE(Jesse): rename this to Cython in Pure Python mode to get Python to quit belly aching.

from libc.math cimport sin, cos, tan, sqrt, pi
#from numpy import sin, cos, tan, sqrt, pi

from numpy import zeros, array, float32, float64, mean, std, int8, uint8, int16, int8, uint16, uint32, int32, uint64, empty, copy, arctan, linspace, nan, absolute
from numpy import sin as numpy_sin

@c.binding(True)
def threshold_sahara(x, a, b, c, d):
    return a * numpy_sin(b * x + c)

@c.binding(True)
def threshold_sahel(x, a, b, c, d):
    return a * numpy_sin(b * x + c) + d #NOTE(Jesse): Notice the extra shift d parameter.

@c.binding(True)
def pointer_sahara(b, c):
    return (-arctan(1 / b) - c) / b

@c.binding(True)
def pointer_sahel(b, c):
    return ((pi/2) - c) / b


@c.exceptval(check=False) #NOTE(Jesse): Cython bug https://github.com/cython/cython/issues/2529
@c.ccall
def get_sahel_model_bounds(samples_max: c.double, samples_min: c.double, number_of_bins: c.ulong) -> tuple:
    bounds_range: c.double = .05
    #NOTE(Jesse): Can the abs and negations be simplified away?
    abs_samples_difference: c.double = abs(samples_max - samples_min)
    a_bounds_min: c.double = -abs_samples_difference - (bounds_range * abs_samples_difference)
    a_bounds_max: c.double = -abs_samples_difference + (bounds_range * abs_samples_difference)

    #NOTE (Eric): Period of the sine model
    two_pi: c.double = 2 * pi
    two_number_of_bins = 2 * number_of_bins

    #NOTE(Jesse): Can this expression be simplified?
    b_bounds_min: c.double = two_pi / two_number_of_bins - (bounds_range * two_pi / two_number_of_bins)
    b_bounds_max: c.double = two_pi / two_number_of_bins + (bounds_range * two_pi / two_number_of_bins)

    #NOTE (Eric): Phase shift of the sine model
    c_bounds_min: c.double = -b_bounds_max
    c_bounds_max: c.double = -b_bounds_min

    #NOTE (Eric): y-intercept
    d_bounds_min: c.double = samples_max - (bounds_range * samples_max)
    d_bounds_max: c.double = samples_max + (bounds_range * samples_max)

    return ([a_bounds_min, b_bounds_min, c_bounds_min, d_bounds_min], [a_bounds_max, b_bounds_max, c_bounds_max, d_bounds_max])

@c.exceptval(check=False) #NOTE(Jesse): Cython bug https://github.com/cython/cython/issues/2529
@c.ccall
def get_sahara_model_bounds(ignore: c.double, ignore2: c.double, ignore3: c.ulong) -> tuple:
    bounds_range: c.double = .05
    a_bounds_min: c.double = 1.12668932 - (bounds_range * 1.12668932)
    a_bounds_max: c.double = 1.12668932 + (bounds_range * 1.12668932)

    b_bounds_min: c.double = 0.21137027 - (bounds_range * 0.21137027)
    b_bounds_max: c.double = 0.21137027 + (bounds_range * 0.21137027)

    c_bounds_min: c.double = 3.53669516 - (bounds_range * 3.53669516)
    c_bounds_max: c.double = 3.53669516 + (bounds_range * 3.53669516)

    d_bounds_min: c.double = 0.0
    d_bounds_max: c.double = 0.0

    return ([a_bounds_min, b_bounds_min, c_bounds_min, d_bounds_min], [a_bounds_max, b_bounds_max, c_bounds_max, d_bounds_max])

#@c.cclass #NOTE(Jesse): Calling this a cclass causes Cython to barf.
class march:
    marching_indices: c.char[:, ::1] = None
    pan_march: c.ushort[::1] = None

    def __init__(self, n_texels: c.long, shadow_direction: c.double[::1]):
        shadow_x: c.float = n_texels * shadow_direction[0]
        shadow_y: c.float = n_texels * shadow_direction[1]

        marching_indices: c.char[:, ::1] = zeros((n_texels, 2), int8)
        for step_number in range(n_texels):
            #NOTE(Jesse): Calculate indices wrt pixel centers.
            marching_indices[step_number][0] = int((shadow_direction[0] * step_number) + 0.5) #NOTE(Jesse): 0.5 for texel center
            marching_indices[step_number][1] = int((shadow_direction[1] * step_number) - 0.5)

        self.marching_indices = marching_indices
        self.pan_march = zeros(n_texels, uint16)

@c.ccall
@c.exceptval(check=False) #NOTE(Jesse): Cython bug https://github.com/cython/cython/issues/2529
def merge_results(vector_tree_layer: object, tmp_files: list) -> c.void:
    import ogr
    ogr.UseExceptions()
    from sys import stderr

    last_percent: c.long = 0

    get_fid = ogr.Feature.GetFID

    get_feature = vector_tree_layer.GetFeature
    set_feature = vector_tree_layer.SetFeature
    vector_tree_layer_layer_defn = vector_tree_layer.GetLayerDefn()

    dst_height_fid: c.long = vector_tree_layer_layer_defn.GetFieldIndex('Height')
    dst_rsme_fid: c.long = vector_tree_layer_layer_defn.GetFieldIndex('rsme')
    dst_shadow_length_fid: c.long = vector_tree_layer_layer_defn.GetFieldIndex('shadow_length')

    tmp_files_count = len(tmp_files)
    for i in range(tmp_files_count):
        tmp_file: str = tmp_files[i]

        split_ds = ogr.Open(tmp_file)
        if split_ds is None:
            print(f'[OGR ERROR] {tmp_file} could not be opened. Skipping')
            continue

        split_mem_ds = ogr.GetDriverByName('Memory').CopyDataSource(split_ds, '')
        if split_mem_ds is None:
            print(f'[ERROR] {tmp_file} could not be opened by Memory dataset. Skipping')
            continue
        split_ds = None

        split_layer = split_mem_ds.GetLayer()
        split_layer.ResetReading()
        split_layer_GetNextFeature = split_layer.GetNextFeature
        split_layer_feature_count: c.long = split_layer.GetFeatureCount()
        print(f'Merging {split_layer_feature_count} features from {tmp_file}.')

        split_layer_defn = split_layer.GetLayerDefn()
        split_height_fid: c.long = split_layer_defn.GetFieldIndex('Height')
        split_rsme_fid: c.long = split_layer_defn.GetFieldIndex('rsme')
        split_shadow_length_fid: c.long = split_layer_defn.GetFieldIndex('shadow_length')

        #NOTE(Jesse): OGR has, behind our backs, changed the STORAGE TYPE, of the fields!!!
        assert(split_layer_defn.GetFieldDefn(split_height_fid).GetType() ==  vector_tree_layer_layer_defn.GetFieldDefn(dst_height_fid).GetType())
        assert(split_layer_defn.GetFieldDefn(split_rsme_fid).GetType() ==  vector_tree_layer_layer_defn.GetFieldDefn(dst_rsme_fid).GetType())
        assert(split_layer_defn.GetFieldDefn(split_shadow_length_fid).GetType() ==  vector_tree_layer_layer_defn.GetFieldDefn(dst_shadow_length_fid).GetType())

        for j in range(split_layer_feature_count):
            split_feature = split_layer_GetNextFeature()
            if split_feature is None:
                print('[ERROR] Should not be none!', file=stderr)
                continue

            src_feature = get_feature(get_fid(split_feature))
            if src_feature is None:
                print('[ERROR] src_feature was None -- should never happen!', file=stderr)
                continue

            split_feature_get_field = split_feature.GetField
            src_feature_height: c.double = split_feature_get_field(split_height_fid)
            src_feature_rsme: c.double = split_feature_get_field(split_rsme_fid)
            src_feature_shadow: c.double = split_feature_get_field(split_shadow_length_fid)

            src_feature_set_field = src_feature.SetField2

            src_feature_set_field(dst_height_fid, src_feature_height)
            src_feature_set_field(dst_rsme_fid, src_feature_rsme)
            src_feature_set_field(dst_shadow_length_fid, src_feature_shadow)
            set_feature(src_feature)

        split_feature = None
        src_feature = None

        split_layer = None
        split_mem_ds = None

        percent_done: c.long = int((float(i + 1) / float(tmp_files_count)) * 100)
        if percent_done != last_percent:
            last_percent = percent_done
            print(percent_done, end=' ', flush=True)
            if percent_done == 100:
                print('Done!')

@c.ccall
#@c.exceptval(check=False) #NOTE(Jesse): Cython bug https://github.com/cython/cython/issues/2529
def determine_tree_heights_internal(payload: tuple, pan_view: c.ushort[:, ::1]) -> c.void:
    #NOTE(Jesse): Cython is having trouble infering the types of integer loop iterators
    i: c.long = 0
    j: c.long = 0
    k: c.long = 0

    import ogr
    ogr.UseExceptions()

    from time import time
    from random import sample
    from scipy.optimize import curve_fit, OptimizeWarning
    from sys import stderr

    #import matplotlib.pyplot as plt

    from numpy import seterr
    seterr(all='raise') #NOTE(Jesse): This will raise an exception if a numpy function encounters a problem (like overflow)

    #NOTE(Jesse): Valid for rev2 mosaic subtiles.
    x_meter_resolution: c.double = 0.5
    #y_meter_resolution: c.double = -0.5

    job_index: c.int = payload[0]
    cutline_index: c.int = payload[1]
    #is_high_density_model: c.bool = payload[2]
    cutline_file: str = payload[2]
    vector_file: str = payload[3]
    required_sample_size: c.int = payload[4]
    trees_extent: tuple = payload[5]
    payload = None

    #print('1')

    vector_disk_ds = ogr.Open(vector_file)
    if vector_disk_ds is None:
        print("[ERROR] Could not create feature OGR GeoPackage vector dataset. Exiting", file=stderr)
        return

    cutline_disk_ds = ogr.Open(cutline_file)
    if cutline_disk_ds is None:
        print("[ERROR] Could not open cutline OGR vector dataset. Exiting", file=stderr)
        return

    out_mem_ds = ogr.GetDriverByName('Memory').CreateDataSource('')
    if out_mem_ds is None:
        print("[ERROR] Could not create feature 'out' OGR GeoPackage memory vector dataset. Exiting", file=stderr)
        return

    cutline_layer = cutline_disk_ds.GetLayer(0)
    cutline = cutline_layer.GetFeature(cutline_index)
    cutline_layer_spatial_ref = cutline_layer.GetSpatialRef()

    temp_ring = ogr.Geometry(ogr.wkbLinearRing)
    temp_ring.AddPoint(trees_extent[0], trees_extent[3])
    temp_ring.AddPoint(trees_extent[1], trees_extent[3])
    temp_ring.AddPoint(trees_extent[1], trees_extent[2])
    temp_ring.AddPoint(trees_extent[0], trees_extent[2])
    temp_ring.AddPoint(trees_extent[0], trees_extent[3])
    trees_layer_extent_geometry = ogr.Geometry(ogr.wkbPolygon)
    trees_layer_extent_geometry.AddGeometry(temp_ring)
    temp_ring = None
    trees_layer_extent_geometry.AssignSpatialReference(cutline_layer_spatial_ref)

    cutline_geo = cutline.GetGeometryRef()

    cutline_geo_count: c.int = cutline_geo.GetGeometryCount() #NOTE(Jesse): Find the largest single cutline polygon to use.  Multiple polygons almost always mean tiny little slivers.
    if cutline_geo_count > 1:
        assert cutline_geo.GetGeometryName() == 'MULTIPOLYGON'

        largest_cutline_polygon_index: c.int = 0

        cutline_geo_area: c.float = 0.0
        for i in range(cutline_geo_count):
            cutline_geo_polygon = cutline_geo.GetGeometryRef(i)
            assert cutline_geo_polygon.GetGeometryName() == 'POLYGON'

            this_cutline_geo_polygon_area: c.float = cutline_geo_polygon.Area()
            if this_cutline_geo_polygon_area > cutline_geo_area:
                largest_cutline_polygon_index = i
                cutline_geo_area = this_cutline_geo_polygon_area

        cutline_geo_polygon = None
        cutline_geo = cutline_geo.GetGeometryRef(largest_cutline_polygon_index)

    cutline_trees_geo = cutline_geo.Intersection(trees_layer_extent_geometry) #NOTE(Jesse): This whole intersection dance here may be unnecessary.
    assert cutline_trees_geo

    out_mem_layer = out_mem_ds.CreateLayer('trees', cutline_layer_spatial_ref, ogr.wkbPolygon)
    assert out_mem_layer

    fields: list = [ogr.FieldDefn("Height", ogr.OFTReal), ogr.FieldDefn("rsme", ogr.OFTReal), ogr.FieldDefn("shadow_length", ogr.OFTReal)]
    for field in fields:
        out_mem_layer.CreateField(field)
    fields = None

    sun_elevation: c.double = cutline.GetFieldAsDouble('SUN_ELEV')
    off_nadir: c.double = cutline.GetFieldAsDouble('OFF_NADIR')
    sun_azimuth: c.double = cutline.GetFieldAsDouble('SUN_AZ')

    shadow_dir_radians: c.double = (90.0 - sun_azimuth + 180) * (pi / 180.0)
    shadow_direction_unitcircle: c.double[::1] = empty(2)
    shadow_direction_raster: c.double[::1] = empty(2)
    shadow_direction_unitcircle[0] = cos(shadow_dir_radians)
    shadow_direction_unitcircle[1] = sin(shadow_dir_radians)

    shadow_direction_raster[0] = shadow_direction_unitcircle[0]
    shadow_direction_raster[1] = -shadow_direction_unitcircle[1] #NOTE(Jesse): Remember y grows DOWN the mosai

    #NOTE(Jesse): The intersection test occurs at the geometry level -- NOT the centroids, so we have to push this in to account for larger features on the edge of the rect.
    shadow_trace_texels: c.long = 24 # NOTE (Eric): I changed the n-value to something a little larger for the non-general model fits. This is to account for tree shadows that may be up to 12 meters c.long
    shadow_trace_meters: c.long  = int(shadow_trace_texels * x_meter_resolution)

    shadow_trace_x_meters: c.double = shadow_trace_meters * shadow_direction_unitcircle[0]
    shadow_trace_y_meters: c.double = shadow_trace_meters * shadow_direction_unitcircle[1]

    shadow_trace_x_texels: c.double = shadow_trace_meters * shadow_direction_raster[0]
    shadow_trace_y_texels: c.double = shadow_trace_meters * shadow_direction_raster[1]

    cutline_trees_line = ogr.Geometry(ogr.wkbLineString)
    cutline_trees_line.AssignSpatialReference(cutline_layer.GetSpatialRef())

    cutline_trees_geo_line = cutline_trees_geo.GetGeometryRef(0)
    assert cutline_trees_geo_line.GetGeometryName() == 'LINEARRING'

    cutline_trees_unique_point_count: c.long = cutline_trees_geo_line.GetPointCount() - 1 #NOTE(Jesse): Remember the first and last are duplicated to form a closed polygon
    for i in range(cutline_trees_unique_point_count):
        line_point: tuple = cutline_trees_geo_line.GetPoint(i) #NOTE(Jesse): You've seen it here first folks. GetPoint is not symmetric with SetPoint!
        cutline_trees_line.AddPoint(line_point[0], line_point[1])
    line_point = cutline_trees_geo_line.GetPoint(0)
    cutline_trees_line.AddPoint(line_point[0], line_point[1])
    line_point = None

    cutline_trees_buffer = cutline_trees_line.Buffer(shadow_trace_meters * 2.2, 2) #NOTE(Jesse): We need to track features within the buffer.
    buffer_mem_ds = ogr.GetDriverByName('Memory').CreateDataSource('')
    if buffer_mem_ds is None:
        print("[ERROR] Could not create feature OGR GeoPackage memory vector dataset. Exiting", file=stderr)
        return

    buffer_trees_disk_layer = vector_disk_ds.ExecuteSQL(f"SELECT fid, Lat, Long, RasterX, RasterY, geom FROM trees", spatialFilter=cutline_trees_buffer)
    buffer_mem_ds.CopyLayer(buffer_trees_disk_layer, 'buffered_cutline_trees')
    buffer_trees_disk_layer = None

    buffered_trees_mem_layer = buffer_mem_ds.GetLayer(0) #NOTE(Jesse): We do not process heights for these, but they are used for intersection tests at the boundaries of cutlines / processing regions.
    buffered_trees_count: c.int = buffered_trees_mem_layer.GetFeatureCount()

    buffered_trees_mem_layer_defn = buffered_trees_mem_layer.GetLayerDefn()

    buffered_feat_lat_fid: c.long = buffered_trees_mem_layer_defn.GetFieldIndex('Lat')
    buffered_feat_long_fid: c.long = buffered_trees_mem_layer_defn.GetFieldIndex('Long')
    buffered_feat_rx_fid: c.long = buffered_trees_mem_layer_defn.GetFieldIndex('RasterX')
    buffered_feat_ry_fid: c.long = buffered_trees_mem_layer_defn.GetFieldIndex('RasterY')

    shrunk_geo = cutline_trees_geo.Difference(cutline_trees_buffer)

    if False: #DEBUG(Jesse): This saves out the subtracted cutline polygon for viewing.
        test_A = ogr.GetDriverByName('GPKG').CreateDataSource('buf_test_{}.gpkg'.format(job_index))
        test_A.CreateLayer('buf', cutline_layer_spatial_ref, ogr.wkbPolygon)
        test_A_layer = test_A.GetLayerByName('buf')

        field = ogr.FieldDefn("empty", ogr.OFTReal)
        test_A_layer.CreateField(field)
        defn = test_A_layer.GetLayerDefn()

        ftr = ogr.Feature(defn)
        #ftr.SetGeometry(shrunk_geo)
        #ftr.SetGeometry(trees_layer_extent_geometry)
        ftr.SetGeometry(cutline_trees_buffer)
        test_A_layer.CreateFeature(ftr)
        ftr = None
        test_A_layer = None
        test_A = None

    trees_layer_extent_geometry = None
    cutline_geo = None
    cutline_trees_lstring = None
    cutline_layer_spatial_ref = None
    cutline = None
    cutline_layer = None
    cutline_disk_ds = None

    trees_mem_ds = ogr.GetDriverByName('Memory').CreateDataSource('')
    if trees_mem_ds is None:
        print("[ERROR] Could not create feature OGR GeoPackage memory vector dataset. Exiting", file=stderr)
        return

    #NOTE(Jesse): I tried to find a way to remove the geometry from the trees layer but gave up after about a dozen different attempts. Nuts this isn't easy.
    shrunk_trees_disk_layer = vector_disk_ds.ExecuteSQL(f"SELECT fid, Area, Lat, Long, RasterX, RasterY, rsme, Height, shadow_length, geom FROM trees", spatialFilter=shrunk_geo)
    trees_mem_ds.CopyLayer(shrunk_trees_disk_layer, 'shrunk_cutline_trees')
    shrunk_trees_disk_layer = None

    vector_disk_ds = None
    cutline_trees_geo = None

    trees_mem_layer = trees_mem_ds.GetLayer(0)
    assert trees_mem_layer

    #print('2')

    trees_mem_layer_defn = trees_mem_layer.GetLayerDefn()
    feat_area_fid: c.long = trees_mem_layer_defn.GetFieldIndex('Area')
    feat_lat_fid: c.long = trees_mem_layer_defn.GetFieldIndex('Lat')
    feat_long_fid: c.long = trees_mem_layer_defn.GetFieldIndex('Long')

    feat_raster_x_fid: c.long = trees_mem_layer_defn.GetFieldIndex('RasterX')
    feat_raster_y_fid: c.long = trees_mem_layer_defn.GetFieldIndex('RasterY')

    feat_feature_height_fid: c.long = trees_mem_layer_defn.GetFieldIndex('Height')
    feat_rsme_fid: c.long = trees_mem_layer_defn.GetFieldIndex('rsme')
    feat_shadow_length_fid: c.long = trees_mem_layer_defn.GetFieldIndex('shadow_length')
    trees_mem_layer_defn = None

    geom_ref = ogr.Feature.GetGeometryRef

    shadow_line_feature_filter = ogr.Geometry(ogr.wkbLineString)
    shadow_line_feature_filter.AddPoint_2D(0.0, 0.0)
    shadow_line_feature_filter.AddPoint_2D(0.0, 0.0)
    shadow_line_set_point = shadow_line_feature_filter.SetPoint_2D
    shadow_line_feature_filter_intersects = shadow_line_feature_filter.Intersects

    trees_mem_layer.ResetReading()
    filtered_trees_count: c.long = trees_mem_layer.GetFeatureCount()
    if filtered_trees_count == 0:
        print(f'[WARN] No trees to process! Skipping cutline {cutline_index}.', file=stderr)
        return

    is_high_density_model: c.bool = (filtered_trees_count / shrunk_geo.Area()) >= 0.0008
    if is_high_density_model:
        print(f'High density model selected for cutline {cutline_index}.')
    else:
        print(f'Low density model selected for cutline {cutline_index}.')

    #NOTE(Jesse): Default to sahara, but change to sahel if necessary.
    pointer = pointer_sahara
    threshold = threshold_sahara
    get_model_bounds = get_sahara_model_bounds
    if is_high_density_model:
        pointer = pointer_sahel
        threshold = threshold_sahel
        get_model_bounds = get_sahel_model_bounds

    print(f'Processing {filtered_trees_count} filtered trees.')
   # before = time()

    #NOTE(Jesse): There were enough 'raw' samples for a valid statistical test. Now 'correct' for the number of raw samples.
    #NOTE(Jesse): https://en.wikipedia.org/wiki/Sample_size_determination
    randomly_selected_features_count: c.long = required_sample_size
    if filtered_trees_count < required_sample_size:
        maximum_sample_size_given_tiny_population: c.long = int(required_sample_size / (1 + ((required_sample_size - 1) / filtered_trees_count)))
        if maximum_sample_size_given_tiny_population < 500: #TODO(Jesse): Smarter number
            print('Cutline does not possess sufficient features after corrected filtering.', file=stderr)
            return

        randomly_selected_features_count = maximum_sample_size_given_tiny_population
    print(f'Gathering {randomly_selected_features_count} features for general model fit.')

    #NOTE(Jesse): pay the OGR function calls cost once upfront. Stuff this data in memory we control.
    trees_mem_layer_GetNextFeature = trees_mem_layer.GetNextFeature

    cutline_trees_geo_ext: c.double[::1] = array(buffered_trees_mem_layer.GetExtent())

    meters_per_bin: c.int = 100
    texels_per_bin: c.int = meters_per_bin * 2
    max_perp_texel_distance: c.int = int(sqrt(texels_per_bin*texels_per_bin + texels_per_bin*texels_per_bin) + 1)  #NOTE(Jesse): This is the furthest distance from the bottom left to the right right corners at 100m^2 in texels.
    meters_per_bin_inv: c.double = float(1) / float(meters_per_bin)
    long_diff: c.int = int(cutline_trees_geo_ext[1] - cutline_trees_geo_ext[0])
    lat_diff: c.int = int(cutline_trees_geo_ext[3] - cutline_trees_geo_ext[2])

    long_bins: c.int = int(long_diff * meters_per_bin_inv) + 1 #NOTE(Jesse): Only when long_diff % meters_per_bin == 0 are there exactly enough bins per area, otherwise we are under counting.  So just add 1 always to account for the remaining space.
    lat_bins: c.int = int(lat_diff * meters_per_bin_inv) + 1

    max_cutline_area_bin_count: c.int = 450
    cutline_area_bins_counts: c.ushort[:, ::1] = zeros((lat_bins, long_bins), uint16) #NOTE(Jesse): These store number of FIDS of features which fall into these bins.
    cutline_area_bins: c.ulonglong[:, :, ::1] = zeros((lat_bins, long_bins, max_cutline_area_bin_count), uint64) #NOTE(Jesse): These store FIDS themselves of features which fall into these bins. Max of 350
    cutline_area_raster_coords: c.ushort[:, :, :, ::1] = zeros((lat_bins, long_bins, max_cutline_area_bin_count, 2), uint16)

    feature_data: c.double[:, ::1] = zeros((filtered_trees_count, 13)) #NOTE(Jesse): FID, lat, long, x, y, height, rsme, shadow length, ftr_lat_to_bin, ftr_long_to_bin, shadow_long_bin, shadow_lat_bin, Area

    get_fid = ogr.Feature.GetFID
    set_fid = ogr.Feature.SetFID

    for i in range(filtered_trees_count):
        ftr = trees_mem_layer_GetNextFeature()
        assert ftr

        ftr_fid: c.ulonglong = get_fid(ftr)

        ftr_getfield = ftr.GetField

        ftr_area: c.float = ftr_getfield(feat_area_fid)

        ftr_raster_x: c.ulong = ftr_getfield(feat_raster_x_fid)
        ftr_raster_y: c.ulong = ftr_getfield(feat_raster_y_fid)

        assert abs(shadow_trace_x_texels) <= ftr_raster_x < (50_000 - abs(shadow_trace_x_texels))
        assert abs(shadow_trace_y_texels) <= ftr_raster_y < (50_000 - abs(shadow_trace_y_texels))

        ftr_lat: c.float = ftr_getfield(feat_lat_fid)
        ftr_long: c.float = ftr_getfield(feat_long_fid)

        ftr_long_to_bin_coord: c.float = ftr_long - cutline_trees_geo_ext[0]
        ftr_lat_to_bin_coord: c.float = ftr_lat - cutline_trees_geo_ext[2]

        assert ftr_long_to_bin_coord > 0.0
        assert ftr_lat_to_bin_coord > 0.0

        ftr_long_to_bin_index: c.long = int(ftr_long_to_bin_coord * meters_per_bin_inv)
        ftr_lat_to_bin_index: c.long = int(ftr_lat_to_bin_coord * meters_per_bin_inv)

        assert ftr_lat_to_bin_index < lat_bins
        assert ftr_long_to_bin_index < long_bins

        cutline_area_bins_count: c.ushort = cutline_area_bins_counts[ftr_lat_to_bin_index][ftr_long_to_bin_index]
        if cutline_area_bins_count >= max_cutline_area_bin_count:
            print('[ERROR] Area bin count exceeds expectation! Exiting.', file=stderr)
            return

        cutline_area_bins[ftr_lat_to_bin_index][ftr_long_to_bin_index][cutline_area_bins_count] = ftr_fid
        cutline_area_raster_coords[ftr_lat_to_bin_index][ftr_long_to_bin_index][cutline_area_bins_count][0] = ftr_raster_x
        cutline_area_raster_coords[ftr_lat_to_bin_index][ftr_long_to_bin_index][cutline_area_bins_count][1] = ftr_raster_y
        cutline_area_bins_counts[ftr_lat_to_bin_index][ftr_long_to_bin_index] += 1

        shadow_lat_to_bin_index: c.long = int((ftr_lat_to_bin_coord + shadow_trace_y_meters) * meters_per_bin_inv)
        shadow_long_to_bin_index: c.long = int((ftr_long_to_bin_coord + shadow_trace_x_meters) * meters_per_bin_inv)

        feature_data[i][0] = ftr_fid
        feature_data[i][1] = ftr_lat
        feature_data[i][2] = ftr_long
        feature_data[i][3] = ftr_raster_x
        feature_data[i][4] = ftr_raster_y
        feature_data[i][5] = ftr_lat_to_bin_index
        feature_data[i][6] = ftr_long_to_bin_index
        feature_data[i][7] = shadow_lat_to_bin_index
        feature_data[i][8] = shadow_long_to_bin_index
        feature_data[i][9] = ftr_area

    buffered_trees_mem_layer_GetNextFeature = buffered_trees_mem_layer.GetNextFeature
    buffered_trees_mem_layer_GetFeature = buffered_trees_mem_layer.GetFeature
    for i in range(buffered_trees_count):
        ftr = buffered_trees_mem_layer_GetNextFeature()
        assert ftr

        ftr_fid: c.ulonglong = get_fid(ftr)

        ftr_getfield = ftr.GetField

        ftr_lat = ftr_getfield(buffered_feat_lat_fid)
        ftr_long = ftr_getfield(buffered_feat_long_fid)

        ftr_raster_x = ftr_getfield(buffered_feat_rx_fid)
        ftr_raster_y = ftr_getfield(buffered_feat_ry_fid)

        ftr_long_to_bin_coord = ftr_long - cutline_trees_geo_ext[0]
        ftr_lat_to_bin_coord = ftr_lat - cutline_trees_geo_ext[2]

        assert ftr_long_to_bin_coord > 0.0
        assert ftr_lat_to_bin_coord > 0.0

        ftr_long_to_bin_index = int(ftr_long_to_bin_coord * meters_per_bin_inv)# + 0.5)
        ftr_lat_to_bin_index = int(ftr_lat_to_bin_coord * meters_per_bin_inv)# + 0.5)

        cutline_area_bins_count = cutline_area_bins_counts[ftr_lat_to_bin_index][ftr_long_to_bin_index]
        if cutline_area_bins_count >= max_cutline_area_bin_count:
            print('[ERROR] Area bin count exceeds expectation! Exiting.', file=stderr)
            return

        cutline_area_bins[ftr_lat_to_bin_index][ftr_long_to_bin_index][cutline_area_bins_count] = ftr_fid
        cutline_area_raster_coords[ftr_lat_to_bin_index][ftr_long_to_bin_index][cutline_area_bins_count][0] = ftr_raster_x
        cutline_area_raster_coords[ftr_lat_to_bin_index][ftr_long_to_bin_index][cutline_area_bins_count][1] = ftr_raster_y
        cutline_area_bins_counts[ftr_lat_to_bin_index][ftr_long_to_bin_index] += 1

    ftr_getfield = None
    ftr = None
    trees_mem_layer.ResetReading()
    trees_mem_layer_get_feature = trees_mem_layer.GetFeature

    #print('3')

    random_feature_indices: c.ulong[::1] = array(sample(range(0, filtered_trees_count), filtered_trees_count), dtype=uint64)
    random_features_xy: c.ushort[:, ::1] = empty((randomly_selected_features_count, 2), uint16)

    random_feature_index: c.long = 0
    intersections_cutline_count: c.long = 0
    #before = time()

    for i in range(filtered_trees_count):
        ftr_area = feature_data[random_feature_indices[i]][9]
        if ftr_area >= 400.0:
            continue

        ftr_lat_to_bin_index = int(feature_data[random_feature_indices[i]][5])
        ftr_long_to_bin_index = int(feature_data[random_feature_indices[i]][6])

        shadow_lat_to_bin_index = int(feature_data[random_feature_indices[i]][7])
        shadow_long_to_bin_index = int(feature_data[random_feature_indices[i]][8])

        ftr_fid = int(feature_data[random_feature_indices[i]][0])
        ftr_lat = feature_data[random_feature_indices[i]][1]
        ftr_long = feature_data[random_feature_indices[i]][2]

        ftr_raster_x = int(feature_data[random_feature_indices[i]][3])
        ftr_raster_y = int(feature_data[random_feature_indices[i]][4])

        area_lat_next_largest_bin: c.int = min(ftr_lat_to_bin_index + 1, lat_bins)
        area_lat_next_smallest_bin: c.int = max(ftr_lat_to_bin_index - 1, 0)
        assert 0 < area_lat_next_largest_bin - area_lat_next_smallest_bin < 3
        lat_to_bin_index: c.int = 0

        area_long_next_largest_bin: c.int = min(ftr_long_to_bin_index + 1, long_bins)
        area_long_next_smallest_bin: c.int = max(ftr_long_to_bin_index - 1, 0)
        assert 0 < area_long_next_largest_bin - area_long_next_smallest_bin < 3
        long_to_bin_index: c.int = 0

        intersection_count: c.ulong = 0
        for lat_to_bin_index in range(area_lat_next_smallest_bin, area_lat_next_largest_bin, 1):
            for long_to_bin_index in range(area_long_next_smallest_bin, area_long_next_largest_bin, 1):
                cutline_area_bins_count = cutline_area_bins_counts[lat_to_bin_index][long_to_bin_index]
                for j in range(cutline_area_bins_count): #TODO(Jesse): Check neighboring bins...then later only spatially relevant cells
                    area_fid: ulonglong = cutline_area_bins[lat_to_bin_index][long_to_bin_index][j]
                    if area_fid == ftr_fid:
                        continue

                    area_raster_x: c.ushort = cutline_area_raster_coords[lat_to_bin_index][long_to_bin_index][j][0]
                    area_raster_y: c.ushort = cutline_area_raster_coords[lat_to_bin_index][long_to_bin_index][j][1]

                    from_nearby_to_ftr_dir_x: c.int = area_raster_x - ftr_raster_x
                    from_nearby_to_ftr_dir_y: c.int = area_raster_y - ftr_raster_y

                    #NOTE(Jesse): Meters are 2x texel size.
                    if abs(from_nearby_to_ftr_dir_x) > abs(shadow_trace_x_texels * 5) or abs(from_nearby_to_ftr_dir_y) > abs(shadow_trace_y_texels * 5):
                        continue

                    texel_distance_square: c.uint = from_nearby_to_ftr_dir_x * from_nearby_to_ftr_dir_x + from_nearby_to_ftr_dir_y * from_nearby_to_ftr_dir_y
                    texel_distance: c.float = sqrt(texel_distance_square)
                    if texel_distance > 1.0:
                        texel_distance_inv: c.float = 1.0 / texel_distance
                        dir_x_norm: c.float = from_nearby_to_ftr_dir_x * texel_distance_inv
                        dir_y_norm: c.float = from_nearby_to_ftr_dir_y * texel_distance_inv

                        if dir_x_norm*shadow_direction_raster[0] + dir_y_norm*shadow_direction_raster[1] < -0.1: #NOTE(Jesse): Filter out features that are not facing in the direction of the shadow.
                            continue

                    shadow_line_set_point(0, ftr_long, ftr_lat)
                    shadow_line_set_point(1, ftr_long + shadow_trace_x_meters, ftr_lat + shadow_trace_y_meters)

                    mem_ftr = trees_mem_layer_get_feature(area_fid) #NOTE(Jesse): Must keep mem_ftr memory live when performing intersection test otherwise georef is invalid.
                    if mem_ftr is None: #NOTE(Jesse): Happens with buffered FIDS!!
                        mem_ftr = buffered_trees_mem_layer_GetFeature(area_fid)
                    assert mem_ftr

                    if shadow_line_feature_filter_intersects(geom_ref(mem_ftr)):
                        intersection_count += 1
                        break
        intersections_cutline_count += intersection_count

        if intersection_count == 0:
            random_features_xy[random_feature_index][0] = ftr_raster_x
            random_features_xy[random_feature_index][1] = ftr_raster_y

            random_feature_index += 1

        if random_feature_index == randomly_selected_features_count:
            break
    else:
        print(f'[WARN] Could not locate {randomly_selected_features_count} random samples for the general model fit.  Skipping cutline {cutline_index}.', file=stderr)
        return

    mem_ftr = None
    #after = time()
    #print(after - before)

    print(f'Skipped {intersections_cutline_count} from intersections.')
    intersections_cutline_count = 0

    #print('4')

    #after = time()
    #print(after - before)

    #NOTE(Jesse): Here we prepare for traversing the raster in any direction.
    #NOTE(Jesse): Before, it was a lot easier.  We just read in 32x32 pixels and traversed in whichever direction we wanted. It worked.
    #             But, it read in 3/4 more data than necessary, so multiply that for every tree the algorithm computes.
    #             Now, we compute the exact bounds to read based on the shadow direction.
    #             You can see doing it this way is really rather tricky to get all the details right.  Maybe I haven't thought of a simpler way.

    #NOTE(Jesse): Figure out which direction the shadows are facing.
    #NOTE(Jesse): Start at north on unit circle: 90D.  We know the sun_azimuth is relative to the north. We just need to know whether to add or subtract relative from that point of origin.
    #Units grow clockwise on the celestrial circle, but grow counter-clockwise  on the unit circle, where our sin / cos are defined -- so subtract!
    #So 90 - sun_azimuth.  That's the direction to the sun.  We want to reflect that across x and y axes to find the direction the sun's rays point, so add (or subtract, doesn't matter since it's symetric) 180d.
    #Now we just need it in radians so do the pi/180 unit conversion and voila!
    primary_gen_traversal: march = march(shadow_trace_texels, shadow_direction_raster)
    marching_indices: c.char[:, ::1] = primary_gen_traversal.marching_indices
    random_features_pan_samples = zeros((shadow_trace_texels), dtype=uint64)
    random_features_pan_samples_view: c.ulonglong[::1] = random_features_pan_samples

    for i in range(randomly_selected_features_count):
        ftr_raster_x = random_features_xy[i][0]
        ftr_raster_y = random_features_xy[i][1]

        for step in range(shadow_trace_texels):
            index_x: c.char = marching_indices[step][0]
            index_y: c.char = marching_indices[step][1]

            #NOTE(Jesse): No data handling?  The odds of having a substantial amount of no data values
            #             around features is slim since feature detection itself requires real data values.

            random_features_pan_samples_view[step] += pan_view[ftr_raster_y + index_y][ftr_raster_x + index_x]

    for i in range(shadow_trace_texels):
        random_features_pan_samples_view[i] /= randomly_selected_features_count

    random_features_pan_samples_std = std(random_features_pan_samples)
    if random_features_pan_samples_std == 0.0:
        print('[ERROR] ALL Pan samples for model fit were identical. STD == 0. Exiting.', file=stderr)
        return

    random_features_pan_samples_standardized = (random_features_pan_samples - mean(random_features_pan_samples)) / random_features_pan_samples_std
    random_features_pan_samples_standardized_view: c.double[::1] = random_features_pan_samples_standardized

    random_feature_sample_steps = linspace(0, shadow_trace_texels, shadow_trace_texels)
    try:
        model_bounds = get_model_bounds(random_features_pan_samples_standardized.max(), random_features_pan_samples_standardized.min(), random_feature_sample_steps.shape[0])
        general_curve_fitted_parameters, _ = curve_fit(threshold, random_feature_sample_steps, random_features_pan_samples_standardized, bounds=model_bounds)

    except (ValueError, RuntimeError, OptimizeWarning) as e:
        print(e)
        print(f"Could not fit general model against cutline index {cutline_index} (probably a cutline with very few features. Setting threshold to a known, arbituary value to test against later.")
        return

    #print(general_curve_fitted_parameters)
    fit_a, fit_b, fit_c, fit_d = general_curve_fitted_parameters

    if fit_b == 0.0:
        print('[ERROR]fit_b parameter is 0.0 by curve_fit and invalid. Exiting.', file=stderr)
        return

    #plt.plot(random_feature_sample_steps, random_features_pan_samples_standardized, 'b-', label='standardized samples')
    #plt.plot(random_feature_sample_steps, threshold(random_feature_sample_steps, fit_a, fit_b, fit_c, fit_d), 'r-', label='fitted sin')
    #plt.xlabel('steps')
    #plt.ylabel('average pan std values')

    x_intersection_point: c.double = pointer(fit_b, fit_c)
    general_threshold_value: c.double = threshold(x_intersection_point, fit_a, fit_b, fit_c, fit_d)
    print(f'Threshold value: {general_threshold_value}')
    print('Done fitting the general model.')

    if general_threshold_value > 0.0:
        print('General threshold value is not negative (it should be because it marks the beginning of the shadow! Exiting.', file=stderr)
        return

    #plt.show()

    pi_over_180: c.double = pi / 180
    sun_elevation_rads: c.double = sun_elevation * pi_over_180
    sin_off_nadir_rads: c.double = sin(off_nadir * pi_over_180)
    off_nadir_adjustment: c.double = 1 / (sqrt(2 * (1 + (sin_off_nadir_rads * sin_off_nadir_rads))))
    tan_sun_rads_off_nadir_adjustment: c.double = tan(sun_elevation_rads) * off_nadir_adjustment  #NOTE(Jesse) This is the multiplier to account for off_nadir distortions in the imagery
    shadow_dir_radians: c.double = (90 - sun_azimuth + 180) * pi_over_180
    left_shift_10_deg: c.double = shadow_dir_radians + 0.174533
    right_shift_10_deg: c.double = shadow_dir_radians - 0.174533

    left_shadow_direction: c.double[::1] = empty(2)
    left_shadow_direction[0] = cos(left_shift_10_deg)
    left_shadow_direction[1] = -sin(left_shift_10_deg)

    right_shadow_direction: c.double[::1] = empty(2)
    right_shadow_direction[0] = cos(right_shift_10_deg)
    right_shadow_direction[1] = -sin(right_shift_10_deg)

    traversals: tuple = (primary_gen_traversal,
                            march(shadow_trace_texels, left_shadow_direction),
                            march(shadow_trace_texels, right_shadow_direction))

    #before = time()
    #print('5')

    #before = time()

    feature_percent_done: c.long = 0
    for i in range(filtered_trees_count): #IMPORTANT(Jesse): Assumes there is a single shadow bec.longing to the tree in the pan arrays.
        ftr_area = feature_data[i][9]
        if ftr_area >= 400.0:
            continue

        ftr_lat_to_bin_index = int(feature_data[i][5])
        ftr_long_to_bin_index = int(feature_data[i][6])

        shadow_lat_to_bin_index = int(feature_data[i][7])
        shadow_long_to_bin_index = int(feature_data[i][8])

        ftr_fid = int(feature_data[i][0])
        ftr_lat = feature_data[i][1]
        ftr_long = feature_data[i][2]

        ftr_raster_x = int(feature_data[i][3])
        ftr_raster_y = int(feature_data[i][4])

        area_lat_next_largest_bin: c.int = min(ftr_lat_to_bin_index + 1, lat_bins)
        area_lat_next_smallest_bin: c.int = max(ftr_lat_to_bin_index - 1, 0)
        assert 0 < area_lat_next_largest_bin - area_lat_next_smallest_bin < 3
        lat_to_bin_index: c.int = 0

        area_long_next_largest_bin: c.int = min(ftr_long_to_bin_index + 1, long_bins)
        area_long_next_smallest_bin: c.int = max(ftr_long_to_bin_index - 1, 0)
        assert 0 < area_long_next_largest_bin - area_long_next_smallest_bin < 3
        long_to_bin_index: c.int = 0

        intersection_count = 0
        for lat_to_bin_index in range(area_lat_next_smallest_bin, area_lat_next_largest_bin, 1):
            for long_to_bin_index in range(area_long_next_smallest_bin, area_long_next_largest_bin, 1):
                cutline_area_bins_count = cutline_area_bins_counts[lat_to_bin_index][long_to_bin_index]
                for j in range(cutline_area_bins_count): #TODO(Jesse): Check neighboring bins...then later only spatially relevant cells
                    area_fid: ulonglong = cutline_area_bins[lat_to_bin_index][long_to_bin_index][j]
                    if area_fid == ftr_fid:
                        continue

                    area_raster_x: c.ushort = cutline_area_raster_coords[lat_to_bin_index][long_to_bin_index][j][0]
                    area_raster_y: c.ushort = cutline_area_raster_coords[lat_to_bin_index][long_to_bin_index][j][1]

                    from_nearby_to_ftr_dir_x: c.int = area_raster_x - ftr_raster_x
                    from_nearby_to_ftr_dir_y: c.int = area_raster_y - ftr_raster_y

                    if abs(from_nearby_to_ftr_dir_x) > abs(shadow_trace_x_texels * 5) or abs(from_nearby_to_ftr_dir_y) > abs(shadow_trace_y_texels * 5):
                        continue

                    texel_distance_square: c.uint = from_nearby_to_ftr_dir_x * from_nearby_to_ftr_dir_x + from_nearby_to_ftr_dir_y * from_nearby_to_ftr_dir_y
                    texel_distance: c.float = sqrt(texel_distance_square)
                    if texel_distance > 1.0:
                        texel_distance_inv: c.float = 1.0 / texel_distance
                        dir_x_norm: c.float = from_nearby_to_ftr_dir_x * texel_distance_inv
                        dir_y_norm: c.float = from_nearby_to_ftr_dir_y * texel_distance_inv

                        if dir_x_norm*shadow_direction_raster[0] + dir_y_norm*shadow_direction_raster[1] < -0.1: #NOTE(Jesse): Filter out features that are not facing in the direction of the shadow.
                            continue

                    shadow_line_set_point(0, ftr_long, ftr_lat)
                    shadow_line_set_point(1, ftr_long + shadow_trace_x_meters, ftr_lat + shadow_trace_y_meters)

                    mem_ftr = trees_mem_layer_get_feature(area_fid) #NOTE(Jesse): Must keep mem_ftr memory live when performing intersection test otherwise georef is invalid.
                    if mem_ftr is None: #NOTE(Jesse): Happens with buffered FIDS!!
                        mem_ftr = buffered_trees_mem_layer_GetFeature(area_fid)
                    assert mem_ftr

                    if shadow_line_feature_filter_intersects(geom_ref(mem_ftr)):
                        intersection_count += 1
                        break
        intersections_cutline_count += intersection_count

        failed_intersection_test_value: c.double = -2.0
        final_height = failed_intersection_test_value
        rsme = failed_intersection_test_value
        shadow_texels_length_max = failed_intersection_test_value

        if intersection_count == 0:
            failed_threshold_test_value: c.double = -1.0
            final_height = failed_threshold_test_value
            shadow_texels_length_max: c.double = failed_threshold_test_value
            rsme = failed_threshold_test_value

            for j in range(3):
                traversal = traversals[j]

                pan_march = traversal.pan_march
                pan_march_view: c.ushort[::1] = pan_march

                marching_indices: c.char[:, ::1] = traversal.marching_indices
                for step in range(shadow_trace_texels):
                    index_x = marching_indices[step][0]
                    index_y = marching_indices[step][1]
                    pan_march_view[step] = pan_view[ftr_raster_y + index_y][ftr_raster_x + index_x]

                pan_march_std = std(pan_march)
                if pan_march_std == 0.0:
                    print('[ERROR] Pan samples were identical. STD == 0. Skipping.', file=stderr)
                    continue

                pan_march_standardized = (pan_march - mean(pan_march)) / pan_march_std
                pan_march_standardized_view: c.double[::1] = pan_march_standardized

                if j == 0: #NOTE(Jesse): RSME test for primary traversal.  Not valid for other shadow directions.
                    difference_squared: c.double = 0.0
                    error: c.double = 0.0
                    for k in range(shadow_trace_texels):
                        difference_squared = random_features_pan_samples_standardized_view[k] - pan_march_standardized_view[k]
                        difference_squared *= difference_squared
                        error += difference_squared
                    error /= shadow_trace_texels
                    rsme = sqrt(error)

                for k in range(shadow_trace_texels):
                    reversed_m: c.long = (shadow_trace_texels - 1) - k
                    if pan_march_standardized_view[reversed_m] <= general_threshold_value:
                        index_x = int(marching_indices[reversed_m][0])
                        index_y = int(marching_indices[reversed_m][1])

                        shadow_texels_length: c.double = sqrt(index_x*index_x + index_y*index_y)
                        if shadow_texels_length > shadow_texels_length_max:
                            shadow_texels_length_max = shadow_texels_length

                        break

            #NOTE(Jesse): We have a choice of what to do if we did not fit.  Maybe record the c.longest distance with the minimum pan value and use that as a guess?
            if shadow_texels_length_max > 0.0:
                final_height = shadow_texels_length_max * x_meter_resolution * tan_sun_rads_off_nadir_adjustment
                assert final_height > 0, final_height

        feature_data[i][10] = final_height #NOTE(Jesse): FID, lat, long, x, y, height, rsme, shadow length
        feature_data[i][11] = rsme
        feature_data[i][12] = shadow_texels_length_max

        percent_done: c.long = int((float(i + 1) / float(filtered_trees_count)) * 100)
        if feature_percent_done != percent_done and percent_done % 10 == 0:
            print(percent_done, end=' ', flush=True)
            feature_percent_done = percent_done
            if percent_done == 100:
                print()

    #after = time()
    #print(after - before)

    mem_ftr = None
    trees_mem_layer = None

    #after = time()
    #print(after - before)

    #print('6')

    print(f'{intersections_cutline_count} features skipped from intersections')

    percentage_skipped_from_intersections: c.long = int((float(intersections_cutline_count) / float(filtered_trees_count)) * 100)
    print(f'{percentage_skipped_from_intersections}% of features skipped.')

    out_mem_layer_create_feature = out_mem_layer.CreateFeature
    out_mem_layer_defn = out_mem_layer.GetLayerDefn()
    new_feature = ogr.Feature

    out_feature_height_fid: c.long = out_mem_layer_defn.GetFieldIndex('Height')
    out_rsme_fid: c.long = out_mem_layer_defn.GetFieldIndex('rsme')
    out_shadow_length_fid: c.long = out_mem_layer_defn.GetFieldIndex('shadow_length')

    validate = ogr.Feature.Validate

    for i in range(filtered_trees_count):
        ftr_fid = int(feature_data[i][0])

        ftr_height: c.double = feature_data[i][10]
        ftr_rsme: c.double = feature_data[i][11]
        ftr_shadow_length: c.double = feature_data[i][12]

        ftr = new_feature(out_mem_layer_defn)
        assert ftr

        feat_setfield = ftr.SetField2

        set_fid(ftr, ftr_fid)
        feat_setfield(out_feature_height_fid, ftr_height)
        feat_setfield(out_rsme_fid, ftr_rsme)
        feat_setfield(out_shadow_length_fid, ftr_shadow_length)

        if validate(ftr) == False:
            raise

        out_mem_layer_create_feature(ftr)

    ftr = None
    out_mem_layer_defn = None
    out_mem_layer = None

    dst_vector_disk_ds = ogr.GetDriverByName('GPKG').CopyDataSource(out_mem_ds, vector_file.replace('features', f'split_{job_index}'))
    if dst_vector_disk_ds == None:
        print('[ERROR] Could not copy memory vector dataset to disk. Exiting.')
        return

    trees_mem_ds = None
    dst_vector_disk_ds = None
