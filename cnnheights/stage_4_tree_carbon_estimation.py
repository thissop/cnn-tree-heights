#NOTE(Jesse): This script is to co-register the tree height database with the tree crown database, that is,
# to match tree shadows with their tree crowns.  Once a match is determined, calculate the tree carbon stock.
# A new feature database consisting of both the tree crowns and shadows will be stored (multipolygon).
# Equations to be provided.

tree_canopy_fp = None
tree_heights_fp = None
tile_cutline_fp = None

if __name__ != "__main__":
    print(f"This script {__name__} must be called directly and not imported to be used as a library.  Early exiting.")
    exit()

def main():
    from os.path import normpath, isdir
    from math import ceil, sqrt, square, radians, cos, sin, pow, tan
    from osgeo import ogr
    ogr.UseExceptions()

    global tree_canopy_fp
    global tree_heights_fp
    global tile_cutline_fp

    tree_canopy_fp = normpath(tree_canopy_fp)
    tree_heights_fp = normpath(tree_heights_fp)
    tile_cutline_fp = normpath(tile_cutline_fp)

    #TODO(Jesse): Sanity check to ensure the file names match the same tile region

    assert isdir(tree_canopy_fp), tree_canopy_fp
    assert isdir(tree_heights_fp), tree_heights_fp
    assert isdir(tile_cutline_fp), tile_cutline_fp

    tree_canopy_disk_ds = ogr.Open(tree_canopy_fp)
    tree_heights_disk_ds = ogr.Open(tree_heights_fp)
    cutline_disk_ds = ogr.Open(tile_cutline_fp)

    tree_canopy_mem_ds = ogr.GetDriverByName("Memory").CopyDataSource(tree_canopy_disk_ds, "")
    tree_canopy_disk_ds = None

    tree_heights_mem_ds = ogr.GetDriverByName("Memory").CopyDataSource(tree_heights_disk_ds, "")
    tree_heights_disk_ds = None

    cutline_mem_ds = ogr.GetDriverByName("Memory").CopyDataSource(cutline_disk_ds, "")
    cutline_disk_ds = None

    assert tree_canopy_mem_ds.GetSpatialRef().ExportToWkb() == tree_heights_mem_ds.GetSpatialRef().ExportToWkb()
    assert cutline_mem_ds.GetSpatialRef().ExportToWkb() == tree_heights_mem_ds.GetSpatialRef().ExportToWkb()

    assert tree_heights_mem_ds.GetLayerCount() == 1, tree_heights_mem_ds.GetLayerCount()

    tree_canopy_lyr = tree_canopy_mem_ds.GetLayer(0)
    tree_heights_lyr = tree_heights_mem_ds.GetLayer(0)
    cutline_lyr = cutline_mem_ds.GetLayer(0)

    tree_height_field_count = tree_heights_lyr.GetFieldCount()
    assert tree_height_field_count <= 1, tree_height_field_count
    if tree_height_field_count == 0:
        tree_heights_lyr.CreateField(ogr.FieldDefn("shadow length", ogr.OFTReal))
    else:
        th_lyr_defn = tree_heights_lyr.GetLayerDefn()
        field_name = th_lyr_defn.GetFieldDefn(0).GetName()
        assert field_name == "shadow length", field_name

    cutline_count = cutline_lyr.GetFeatureCount()
    if cutline_count == 0:
        print(f"[ERROR] cutline {tile_cutline_fp} has no geometries! Early exit.")
        return

    canopy_count = tree_canopy_lyr.GetFeatureCount()
    if canopy_count == 0:
        print(f"No canopies in {tree_canopy_fp}.  Early Exit")
        return

    heights_count = tree_heights_lyr.GetFeatureCount()
    if heights_count == 0:
        print(f"No heights in {tree_heights_fp}.  Early Exit")
        return

    #TODO(Jesse): This is currently a painfully linear algorithm.  Multithread.
    cutline_lyr.ResetReading()
    tree_canopy_lyr.ResetReading()
    tree_heights_lyr.ResetReading()
    shadow_no_pair_count = 0

    def is_oblong(envelope):
        (x_min, x_max, y_min, y_max) = *envelope
        x_span = x_max - x_min
        y_span = y_max - y_min

        if abs(x_span / y_span) > 1.5:
            return True

        return False

    #NOTE(Jesse): Via Allometric equations to estimate the dry mass of Sahel woody plants from very-high resolution satellite imagery
    # Table 7 OLS log log with Baskerville correction
    compute_leaf_carbon_stock = lambda area_times_height: 0.2644 * pow(area_times_height, 0.6665)
    compute_wood_carbon_stock = lambda area_times_height: 1.9147 * pow(area_times_height, 0.9017)
    compute_root_carbon_stock = lambda area_times_height: 0.9800 * pow(area_times_height, 0.8263)

    use_spatial_bin_algorithm = False
    if use_spatial_bin_algorithm:
        bins_xy_meters = 100
        for i, c_ftr in enumerate(cutline_lyr):
            c_geo = c_ftr.GetGeometryRef()
            if not c_geo.IsValid():
                #TODO(Jesse): Investigate how to robustly recover the geometry via MakeValid
                continue

            if c_geo.Area() < 300:
                continue #NOTE(Jesse): Arbiturary threshold to skip small cutline geometries

            #NOTE(Jesse): Find the closest tree crown to associate a given height estimate with.
            # Since this must happen per height estimate, for all such tree crowns, this is O(N*M).
            # And, since N ~ M in size, we can think of it as O(N^2).
            # However, we know that tree crowns / heights are spatially coherent.  We only need to check
            # "nearby" trees.  So pre-sort them into smaller spatial bins. (In fact, the tiles themselves are already such an optimization).
            # And then only check the smaller bins for matches.  So, reduce the time complexity to O(N) + O(M*N/bins)
            (c_x_min, c_x_max, c_y_min, c_y_max) = c_geo.GetEnvelope()
            y_span = c_y_max - c_y_min
            x_span = c_x_max - c_x_min

            y_bins_count = int(ceil(y_span / bins_xy_meters))
            x_bins_count = int(ceil(x_span / bins_xy_meters))

            spatial_bins = [[None] * x_bins_count] * y_bins_count

            #IMPORTANT(Jesse): SetSpatialFilter works by only filtering out feature geometries whose
            # extent to not intersect with the spatial filter.  This means that a feature geometry can be
            # returned under multiple spatial filters even if they are disjoint.  This is true for shadows
            # or canopies that cross a cutline boundary.

            tree_canopy_lyr.SetSpatialFilter(c_geo)
            tree_heights_lyr.SetSpatialFilter(c_geo)

            for ftr in tree_canopy_lyr:
                f_geo = ftr.GetGeometryRef()
                (f_x_min, f_x_max, f_y_min, f_y_max) = f_geo.GetEnvelope()

                f_y_center = f_y_min + (f_y_max - f_y_min)
                f_x_center = f_x_min + (f_x_max - f_x_min)

                y_bin = int(ceil(f_y_center / bins_xy_meters))
                x_bin = int(ceil(f_x_center / bins_xy_meters))

                spatial_bins[y_bin][x_bin].append(ftr, f_y_center, f_x_center)

            for h_ftr in tree_heights_lyr:
                h_geo = h_ftr.GetGeometryRef()
                (h_x_min, h_x_max, h_y_min, h_y_max) = h_geo.GetEnvelope()

                h_y_center = h_y_min + (h_y_max - h_y_min)
                h_x_center = h_x_min + (h_x_max - h_x_min)

                y_bin = int(ceil(h_y_center / bins_xy_meters))
                x_bin = int(ceil(h_x_center / bins_xy_meters))

                spatial_bin = spatial_bins[y_bin][x_bin]
                closest_distance = 9999999.9
                closest_ftr = None
                for ftr, f_y_center, f_x_center in spatial_bin:
                    distance = sqrt(square(f_y_center - h_y_center) + square(f_x_center - h_x_center))
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_ftr = ftr

                if closest_ftr:
                    1
    else:
        for i, c_ftr in enumerate(cutline_lyr):
            c_geo = c_ftr.GetGeometryRef()
            if not c_geo.IsValid():
                #TODO(Jesse): Investigate how to robustly recover the geometry via MakeValid
                print(f"cutline geo fid {c_ftr.GetFID()} is invalid, skipping")
                continue

            if c_geo.Area() < 300:
                print(f"cutline geo fid {c_ftr.GetFID()} is too small (area < 300), skipping")
                continue #NOTE(Jesse): Arbiturary threshold to skip small cutline geometries

            tc_lyr = tree_canopy_mem_ds.ExecuteSQL("select * from trees", spatialFilter=c_geo)
            ts_lyr = tree_heights_mem_ds.ExecuteSQL("select * from shadows", spatialFilter=c_geo)

            sun_azimuth = c_ftr.GetFieldAsDouble("SUN_AZ")
            sun_elevation = c_ftr.GetFieldAsDouble("SUN_ELEV")

            solar_flux_angle = sun_azimuth - 180.0
            solar_flux_perpendicular_angle = solar_flux_angle - 90.0
            solar_flux_perp_rad_converted_from_geospatial = radians(90.0 - solar_flux_perpendicular_angle)
            solar_perp_x = cos(solar_flux_perp_rad_converted_from_geospatial)
            solar_perp_y = sin(solar_flux_perp_rad_converted_from_geospatial)

            for ts_ftr in ts_lyr:
                shadow_geo = ts_ftr.GetGeometryRef()
                tc_lyr.SetSpatialFilter(shadow_geo)

                tree_crown_canidate_count = tc_lyr.GetFeatureCount()
                if tree_crown_canidate_count == 0:
                    print(f"[NOTE] shadow {ts_ftr.GetFID()} did not intersect with a tree crown.")
                    shadow_no_pair_count += 1
                    continue

                #TODO(Jesse): If there are ~too many~ intersections then this shadow is probably
                # in a closed canopy region.  Handle this condition better.
                #VITAL(JESSE): This problem is central to our robustness concerns!!!
                #VITAL(JESSE): This problem is central to our robustness concerns!!!
                #VITAL(JESSE): This problem is central to our robustness concerns!!!
                #VITAL(JESSE): This problem is central to our robustness concerns!!!
                #
                #
                #NOTE(Jesse): Recall, shadow lengths are only vital IF the shadow is unobstructed!!
                if tree_crown_canidate_count > 2:
                    continue

                #TODO(Jesse): Choose best option if more than 1 is available
                matching_tc = tc_lyr.GetFeature(0)
                tc_geo = matching_tc.GetGeometryRef()
                if tc_geo.GetArea() > 500: #TODO(Jesse): Handle large tree crowns robustly
                    continue

                #NOTE(Jesse): I do not know if geometry envelopes are stored alongside geometries
                # or if they are rebuilt lazily as needed.  If lazy, then I suspect the GetField()
                # approach is faster, otherwise, computing the envelope center will be for tree centroid
                # determination

                if is_oblong(tc_geo.GetEnvelope()): #TODO(Jesse): Handle oblong tree crowns robustly
                    continue

                shadow_length = 0.0

                #TODO(Jesse): Bias final length selection towards shadow vertices whose angle with the tree crown centroid
                # produces a small dot product wrt the solar flux direction.

                tree_centroid_x = matching_tc.GetFieldAsDouble("Long")
                tree_centroid_y = matching_tc.GetFieldAsDouble("Lat")

                shadow_points_xy = shadow_geo.GetPoints()
                for xy in shadow_points_xy:
                    x = xy[0] - tree_centroid_x
                    y = xy[1] - tree_centroid_y

                    #distance between vertex point and the line that is perpendicular to the solar angle and cuts through the centroid of the tree
                    length = (x * solar_perp_x) + (y * solar_perp_y)
                    if length > shadow_length:
                        shadow_length = length

                tree_crown_height = shadow_length * tan(radians(sun_elevation))
                tree_crown_area = matching_tc.GetFieldAsDouble("Area")

                tc_area_x_height = tree_crown_area * tree_crown_height
                leaf_carbon_estimation = compute_leaf_carbon_stock(tc_area_x_height)
                wood_carbon_estimation = compute_wood_carbon_stock(tc_area_x_height)
                root_carbon_estimation = compute_root_carbon_stock(tc_area_x_height)
                tree_carbon_estimate_total = leaf_carbon_estimation + wood_carbon_estimation + root_carbon_estimation

                #TODO(Jesse): Store results

            tree_canopy_mem_ds.ReleaseResultSet()
            tree_heights_mem_ds.ReleaseResultSet()

main()
