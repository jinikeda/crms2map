#!/usr/bin/env python
# coding: utf-8
# CRMS2Interpolation with error analysis
# Developed by the Center for Computation & Technology and Center for Coastal Ecosystem Design Studio at Louisiana State University (LSU).
# Developer: Jin Ikeda, Shu Gao, Linoj V. N. Rugminiamma, and Christopher E. Kees
# Last modified Feb 17, 2025


from src.CRMS_general_functions import *
from datetime import timedelta

# Calculate the end year (today - 4 months) for default_range (CRMS data are behind around 4 month)
end_year = (datetime.today() - timedelta(days=120)).year
default_range = f"2006_{end_year}"  # CRMS data are available after 2006 to around 4 months before today


def get_CRMS_file_variable(variable):
    """Return the CRMS file name based on the selected variable or abort if invalid."""
    input_filename = {
        'Salinity': "CRMS_Surface_salinity",
        'WL': "CRMS_Geoid99_to_Geoid12b_offsets",
        'Temp': "CRMS_Water_Temp.csv",
        'W_HP': "CRMS_Water_Elevation_to_Marsh",
        'W_depth': "CRMS_Water_Elevation_to_Marsh"
    }

    # 'W_HP' and 'W_depth' add additional name later

    if variable not in input_filename:
        print(f"Error: Variable '{variable}' is not recognized. Please choose from {list(input_filename.keys())}.")
        sys.exit(1)  # Abort execution

    return input_filename[variable]


@click.command()
@click.option(
    "--data_range",
    default=default_range,
    help=(
            f"Start and End year of the datasets (format: YYYY_YYYY). "
            f"Default: 2006_{end_year} (around 4 months before today)."
    ),
    show_default=True,
)
@click.option(
    "--sdate",
    default="2008-01-01",
    help="State date for the datasets (format: YYYY-MM-DD)",
)
@click.option(
    "--edate",
    default="2008-12-31",
    help="End date for the datasets (format: YYYY-MM-DD)",
)
@click.option(
    "--data_type",
    default="M",
    help="Data type: hourly(H), daily(D), monthly(M), and yearly(Y)",
)
@click.option(
    "--tstep",
    default=1,
    type=click.IntRange(1, 1000, clamp=True),
    help=(
            "Numeric time step for the datasets based on '--data_type'.\n"
            "  - If hourly (H): Number of hours (e.g., 1, 3, 6, 12)\n"
            "  - If daily (D): Number of days (e.g., 1, 7, 30)\n"
            "  - If monthly (M): Number of months (e.g., 1, 3, 6, 12)\n"
            "  - If yearly (Y): Number of years (e.g., 1, 5, 10)\n"
            "Default: 1"
    ),
    show_default=True,
    required=False,
)
@click.option(
    "--data_var",
    default='Salinity',
    help="Specify the dependent variable (e.g., Salinity, WL, W_HP, W_depth, Temp).",
)
@click.option(
    "--method",
    default=3,
    type=click.IntRange(1, 4),
    help=(
            "Select Interpolation Method:\n"
            "  1: Kriging using Pykrige\n"
            "  2: IDW with fixed distance (WhiteboxTools, optional: need to install WhiteboxTools))\n"
            "  3: IDW with KDTree (k-nearest neighbors)\n"
            "  4: Random Forest Spatial Interpolation with KDTree"),
)
@click.option(
    "--KNN",
    default=6,
    type=click.IntRange(1, 12),
    help="Number of nearest neighbors (recommend 4-6 for the best performance with IDW)",
)
@click.option(
    "--inputfile",
    type=click.Path(exists=True),
    default=None,
    help=(
            "Alternative input file path (manual input). "
            "This option is not typically required but can be used for manual file selection."
    ),
    required=False,
)
@click.option(
    "--geoinfo",
    default=False,
    help=(
            "For research purposes only (Not Recommend): Include geographic information during Random forest interpolation (method=4)."
            "This approach may enhance spatial accuracy when KNN = 1 or 2 but is not required for standard analysis."
            "This approach also requires a DEM grid, SPI, and Tave files in the Input folder. (For the details, contact us for more information)."

    ),
    required=False,
)
def interpolate_subcommand(data_range, sdate, edate, data_type, tstep, data_var, method, knn, inputfile, geoinfo):
    """Handle interpolation of point-based (station-based) data."""

    ### Step 3 #############################################################################################################
    print('Step 3: Interpolation')
    ########################################################################################################################
    # Import the general interpolation functions from the CRMS_general_functions.py file (some interpolation functions are keep inside of this code)

    start_time = time.time()

    # The target working directory
    ### HPC ####
    # path_folder = "/work/username/ETC/CRMS2Plot/" # if you want to specify the location
    Workspace = os.getcwd()

    # Print the current working directory
    print("Current working directory: {0}".format(os.getcwd()))

    # Change the current working directory
    os.chdir(Workspace)

    ### 3.2 Open reference file

    Inputspace = os.path.join(Workspace, 'Input')  # Make Input folder
    Outputspace = os.path.join(Workspace, 'Output')  # Make Output folder

    try:
        # os.makedirs(Photpspace, exist_ok=True)
        os.makedirs(Inputspace, exist_ok=True)
        os.makedirs(Outputspace, exist_ok=True)

    except Exception as e:
        print(f"An error occurred while creating directories: {e}")

    dtype_str = data_type + "data"  # to add the data type to adjust the input file style
    print(f"Data type is {data_type}")

    Input_subset = get_CRMS_file_variable(data_var)

    if inputfile is not None:
        Input_file = inputfile
    elif inputfile is None and data_var == 'W_HP':
        Input_file = Input_subset + "_" + data_range + "_wd_" + dtype_str + ".csv"
    elif inputfile is None and data_var == 'W_depth':
        Input_file = Input_subset + "_" + data_range + "_wdepth_" + dtype_str + ".csv"
    else:
        Input_file = Input_subset + "_" + data_range + "_" + dtype_str + ".csv"

    print(f"Input file is {Input_file}")

    # Assert check for file existence

    input_file = os.path.join(Inputspace, Input_file)

    if not os.path.exists(input_file):
        print(f"Error: Input file '{Input_file}' not found in '{Inputspace}'.")
        print(f"Check if the data range '{data_range}' is correct.")
        sys.exit(1)  # Abort execution
    else:
        print(f"Input file '{Input_file}' found.")

    ########################################################################################################################
    # Input_file
    ########################################################################################################################
    # FYR: The input file name
    # User can also manually change the input file name here
    # Input_file = "CRMS_Geoid99_to_Geoid12b_offsets_Mdata.csv"  # "CRMS_Surface_salinity_2006_2022_Mdata.csv"

    ########################################################################################################################
    # Input Interpolation method
    ########################################################################################################################
    # Method 1: Kriging using Pykrige
    # Method 2: Inverse distance weight (IDW) with fixed distance using Whitebox (optional: need to install WhiteboxTools)
    # Method 3: Inverse distance weight (IDW) with KDTree (k-nearest neighbors)
    # Method 4: Random Forest Spatial Interpolation with KDTree (k-nearest neighbors)
    ########################################################################################################################
    # Interpolation method
    Method = method
    # geoinfo = geoinfo  # True: Use only method RF for geoinformation, False: use only lat and lon
    basin_info = False  # True: Use basin information, False: use only lat and lon (this is optional)

    # if Method != 2:  # Method 2 is IDW with fixed radius
    #     knn = knn  # k-nearest neighbors'
    # else:
    #     radius = 1.00  # fixed radius
    if Method == 2:  # Method 2 is IDW with fixed radius
        radius = 1.00  # fixed radius

    # Error checker for geoinfo and basin_info
    if Method != 4:  # Method 4 is Random Forest and avoid to use geo and basin information
        geoinfo = False
        basin_info = False

    if geoinfo == False and basin_info == True:
        basin_info = False

    # Randomly pick up dataset
    grid_space = 5  # grid space for random sampling 1/1000 degrees
    #######################################################################################################################
    if Method == 3:  # Method 3 is IDW with KDTree
        power = 2  # power of weight function (default)
    else:
        power = 0  # power of weight function for RF: Generally, increase the power decreases the performance of the model
    #######################################################################################################################
    leafsize = 10  # (default) For KDTree

    # coordinates system
    GCS = "EPSG:4269"  # NAD83
    PRJ = "EPSG:6344"  # NAD83(2011) UTM Zone 15N
    GCS_name_list = ['Long', 'Lat']
    xy_list = ['x', 'y']

    ####################################################################################################################
    #### 3.1 Import modules ###
    ####################################################################################################################

    # Import additional modules
    from rasterio.mask import mask
    from pykrige.ok import OrdinaryKriging
    from src.KDTree_idw import Invdisttree  # Need KDTree_idw.py
    from scipy.spatial import cKDTree as KDTree
    from sklearn.ensemble import RandomForestRegressor

    ####################################################################################################################

    # Get y_variable
    y_variable = get_y_variable(Input_subset)

    assert 1 <= Method <= 4, "Please select the method values between 1-4 ...{{ (>_<) }}\n"

    method_dict = {1: "Krige", 2: "IDWr", 3: "IDWk", 4: "RF"}
    method_str = method_dict.get(Method, "TBD")

    if Method == 2:
        Output_name = f"{y_variable}/Output_{y_variable}_r{radius}_{method_str}"
    else:
        Output_name = f"{y_variable}/Output_{y_variable}_n{knn}_{method_str}"

    if geoinfo == True:
        Output_name = Output_name + "_geo"

    # Output_name = f"Output_b_test"

    print('Interpolation method is \t',
          f"{method_str} with {'fixed radius=' + str(radius) if Method == 2 else 'k-nearest neighbors=' + str(knn)}")

    Output_dir = os.path.join(Outputspace, Output_name)  # Make Output folder
    try:
        os.makedirs(Output_dir, exist_ok=True)
    except Exception as e:
        print(f"An error occurred while creating directories: {e}")

    ########################################################################################################################
    #### Internal functions
    ########################################################################################################################
    def gdal_tiff(z_interp, tiff_name):
        # Make a tiff file
        Raster_tif = os.path.join(Output_dir, tiff_name)
        gtiff_driver = gdal.GetDriverByName('GTiff')  # Use GeoTIFF driver

        out_ds = gtiff_driver.Create(Raster_tif, z_interp.shape[1], z_interp.shape[0], 1,
                                     gdal.GDT_Float64)  # Create a output file
        out_ds.SetProjection(prj.ExportToWkt())
        transform = [xmin, space / scale_factor, 0.0, ymax, 0.0, -space / scale_factor]

        # transform[0] # Origin x coordinate
        # transform[1] # Pixel width
        # transform[2] # x pixel rotation (0° if image is north up)
        # transform[3] # Origin y coordinate
        # transform[4] # y pixel rotation (0° if image is north up)
        # transform[5] # Pixel height (negative)

        out_ds.SetProjection(prj.ExportToWkt())
        transform = [xmin, space / scale_factor, 0.0, ymax, 0.0, -space / scale_factor]

        out_ds.SetGeoTransform(transform)
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(np.flipud(z_interp))  # bottom row <-> top row due to origin of raster file
        out_band = out_ds.GetRasterBand(1).SetNoDataValue(nodata_value)  # Exclude nodata value
        out_band = out_ds.GetRasterBand(1).ComputeStatistics(
            True)  # Calculate statistics for Raster pyramids (Pyramids can speed up the display of raster data)
        print('maximum value of interpolated data is', np.max(z_interp))

        del out_ds
        return Raster_tif

    def idwf_interpolocation(data_train, date, gridx, gridy):  # IDW with fixed radius interpolation

        Base_tif = os.path.join(Inputspace,
                                "PyKrige_process.tif")  # Cell_size in the whitebox will be ignored when we use this option
        IDWRaster_tif = os.path.join(Output_dir, "IDW_process.tif")
        IDWb = wbt.idw_interpolation(i=data_train, field=date, output=IDWRaster_tif, use_z=False, weight=2.0,
                                     radius=1.00,
                                     min_points=8, cell_size=0.005, base=Base_tif)
        Edit_raster = gdal.Open(IDWRaster_tif, 1)  # add projection and
        Edit_raster.SetProjection(prj.ExportToWkt())
        band = Edit_raster.GetRasterBand(1)
        out_band = band.ComputeStatistics(0)  # calculate statistics
        del Edit_raster

        return IDWRaster_tif

    def kriging_interpolation(data_train, date, gridx, gridy, knn):
        ### parameters of OrdinaryKriging
        variogram_model = 'spherical'
        nlags = 6
        weight = True
        verbose = False
        enable_plotting = False
        n_closest_points = knn
        nodata_value = -99999

        x, y, z = data_train[GCS_name_list[0]].values, data_train[GCS_name_list[1]].values, data_train[date].values

        # Spatial interpolation using ordinal kriging
        OK = OrdinaryKriging(x, y, z, variogram_model=variogram_model, nlags=nlags, weight=weight, verbose=verbose,
                             enable_plotting=enable_plotting, coordinates_type="geographic")
        z_interp, ss = OK.execute('grid', gridx, gridy, mask=shapes, backend='C',
                                  n_closest_points=n_closest_points)  # Use cython to speed up the calculation.

        KrigeRaster_tif = gdal_tiff(z_interp, "PyKrige_process.tif")

        return KrigeRaster_tif

    def extract_point_values(raster_path, points_path):
        # Load the shapefile of points
        points = gpd.read_file(points_path)
        # Load the DEM raster
        dem = rasterio.open(raster_path)

        # extract xy from point geometry
        raster_values = []
        array = dem.read(1)
        for point in points.geometry:
            # print(point.xy[0][0],point.xy[1][0])
            x = point.xy[0][0]
            y = point.xy[1][0]
            row, col = dem.index(x, y)

            # Append the z value to the list of z values
            raster_values.append(array[row, col])

            # print("Point correspond to row, col: %d, %d"%(row,col))
            # print(array[row, col])
            # print("Raster value on point %.2f \n"%dem.read(1)[row,col])

        points['interp'] = raster_values
        points.to_file(points_path, driver='ESRI Shapefile')
        del points

        return raster_values

    def get_nearest_variable_index(target_gdf, reference_gdf, str_variable, scale_factor2, leafsize=10,
                                   xy_list=['x', 'y']):
        # Use a projected system
        xy_reference = reference_gdf[xy_list].to_numpy()
        xy_reference = xy_reference / scale_factor2

        xy_target = target_gdf[xy_list].to_numpy()
        xy_target = xy_target / scale_factor2

        dist_tree = KDTree(xy_reference, leafsize=leafsize)
        _, ix = dist_tree.query(xy_target, 1)  # find the nearest neighbor for each point

        # Get the index of the 'Long' column
        loc = target_gdf.columns.get_loc(GCS_name_list[1]) + 1  # after 'Lat' column

        # Insert the new column at the specified location
        target_gdf.insert(loc, str_variable + '_index', ix)

        return target_gdf

    def process_nClimGrid_data(str_variable, nClimGrid_File, GCS, GCS_name_list, PRJ, Grid_prj_gdf, scale_factor2,
                               leafsize,
                               xy_list, str_ref_variable, grid_df, polygon, Output_dir):
        # Create DataFrame from nClimGrid file
        nClimGrid_df = create_dataframe(nClimGrid_File)

        # Get the slice index
        slice_index = nClimGrid_df.columns.get_loc(GCS_name_list[0]) + 1

        # Create point shape file for train data
        nClimGrid_GCS_gdf = create_df2gdf(nClimGrid_df.iloc[:, slice(slice_index)], GCS_name_list, [], GCS,
                                          output_file=None)

        # Convert the coordinates
        nClimGrid_prj_gdf = convert_gcs2coordinates(nClimGrid_GCS_gdf, PRJ)

        # Get the nearest variable index
        Grid_nClimGrid_index = get_nearest_variable_index(Grid_prj_gdf, nClimGrid_prj_gdf, str_variable, scale_factor2,
                                                          leafsize=leafsize, xy_list=xy_list)

        # Assign the variable values into the grid
        loc2 = grid_df.columns.get_loc(str_ref_variable) + 1
        grid_df.insert(loc2, str_variable + '_index', Grid_nClimGrid_index[str_variable + '_index'])

        # Add basin information
        grid_df = Spatial_join_basin(grid_df, polygon, GCS, grid_df.columns.values.tolist())
        grid_df = grid_df.drop('geometry', axis=1)

        # Save the DataFrame to a CSV file
        grid_df.to_csv(os.path.join(Output_dir, f"Grid_{str_variable}_index.csv"), index=True)

        # Save the grid data dummy for the next step in the RF regression model
        grid_df_index = grid_df.copy()

        return grid_df_index, nClimGrid_prj_gdf, nClimGrid_df

    def assign_variable_index2values(target_df, reference_df, date_column, str_target_variable, str_variable):
        # Check if date_column exists in reference_df
        if date_column not in reference_df.columns:
            raise ValueError(f"{date_column} does not exist in reference_df")

        # Check if the indices in target_df[str_variable + '_index'] are within the range of reference_df's index
        if not set(target_df[str_variable + '_index']).issubset(set(reference_df.index)):
            raise ValueError("Some indices in target_df are not found in reference_df")

        # Get the index of the 'Long' column
        target_df_values = target_df.copy()
        loc = target_df.columns.get_loc(
            str_target_variable) + 1  # after 'Lat', and DEM column # need to caution if the column content is changed
        target_df_values.insert(loc, str_variable,
                                reference_df.loc[target_df_values[str_variable + '_index'], date_column].values)
        # target_df_values.to_csv(os.path.join(Output_dir, "target_df.csv"))  # Confirmed the output

        target_df_values.drop([str_variable + '_index'], axis=1, inplace=True)
        return target_df_values

    def calculate_weighted_distance(df_combined, df_train, other_train_list, knn, power):
        wdist = df_combined.iloc[:, knn:2 * knn].values / (df_combined.iloc[:, 0:knn].values ** power)
        w2dists = [(df_train[j].values - df_combined.iloc[:, (i + 2) * knn: (i + 3) * knn].values) / (
                df_combined.iloc[:, 0:knn].values ** power) for i, j in enumerate(other_train_list)]
        w2dists_cols = [
            [f'w{i + 2}dist_{i}' for i in range(1, df_combined.shape[1])] if len(xy) == df_train.shape[0] else [
                f'w{i + 2}dist_{i}' for i in range(1, df_combined.shape[1] + 1)] for _ in other_train_list]
        return wdist, w2dists, w2dists_cols

    def create_rf_table(df_train, df_test, wdist, w2dists, w2dists_cols, date):
        wdist_df = pd.DataFrame(wdist, columns=[f'wdist_{i}' for i in range(1, df_train.shape[1])])
        w2dist_df = pd.DataFrame(w2dists, columns=w2dists_cols)
        rf_table = pd.concat([df_train.iloc[:, slice(df_train.columns.get_loc(date) + 1)], wdist_df, w2dist_df],
                             axis=1) if len(xy) == df_train.shape[0] else pd.concat([df_test, wdist_df, w2dist_df],
                                                                                    axis=1)
        return rf_table

    def create_preRF_df(dist, ix, df_train, z_train_list, N, Train):
        if Train:
            range_value = range(N)
            drop_list = ['dist_0', 'sta_0'] + [f'z{jj}_0' for jj in range(len(z_train_list))]  # drop the last z value
        else:  # Test data
            N = knn
            range_value = range(1, N + 1)
            drop_list = []

        dist_df = pd.DataFrame(dist, columns=[f'dist_{i}' for i in range_value])  # distance
        station_df = pd.DataFrame(df_train.loc[ix.ravel(), 'CRMS_Sta'].values.reshape(ix.shape),
                                  columns=[f'sta_{i}' for i in range_value])
        # Create a DataFrame for each z variable
        z_dfs = [pd.DataFrame(df_train[z_train].values[ix.ravel()].reshape(ix.shape),
                              columns=[f'z{jj}_{i}' for i in range_value]) for jj, z_train in enumerate(z_train_list)]
        df_combined = pd.concat([dist_df, *z_dfs, station_df], axis=1)
        df_combined.drop(drop_list, axis=1, inplace=True)

        return df_combined

    def process_nearest_neighbors_RF(dist_tree, xy, df_train, z_train_list, knn, power, date, Train=True, df_test=None):
        df_z_trains = df_train[z_train_list]  # z values of train data

        if Train:
            if len(xy) != df_z_trains.shape[0]:
                raise ValueError("xy and z must have the same size")
            N = knn + 1  # +1 for the distance to the nearest neighbor because the distance to the nearest neighbor is 0 in train data
            output = os.path.join(Output_dir, "df_combined_train.csv")  # Save merged CSV file
        else:  # Test data
            if df_test is None:
                raise ValueError("df_test must be provided when Train is False")
            N = knn
            output = os.path.join(Output_dir, "df_combined_test.csv")  # Save merged CSV file

        dist, ix = dist_tree.query(xy, N)  # this is for test data
        df_combined = create_preRF_df(dist, ix, df_train, z_train_list, N, Train)
        if Train:
            print(f'average distance of train datasests to nearest neighbors [100 km]: ', np.mean(dist))
        else:
            print(f'average distance of test datasests to nearest neighbors [100 km]: ', np.mean(dist))
        # df_combined.to_csv(output)  # Save merged CSV file comfirm the output

        # Calculate weighted distance
        wdist = df_combined.iloc[:, knn:2 * knn].values / (
                df_combined.iloc[:, 0:knn].values ** power)  # neighbor's target variable*weighted distance
        wdist_df = pd.DataFrame(wdist, columns=[f'wdist_{i + 1}' for i in range(knn)])  # no wdist_0
        w2dist_df = pd.DataFrame()  # set empty DataFrame as a default value

        if date in z_train_list:
            other_train_list = z_train_list.copy()
            other_train_list.remove(date)
            # print('num_other_variables =', len(other_train_list))
            if other_train_list != []:
                w2dists = []
                w2dists_cols = []
                for ii, jj in enumerate(other_train_list):
                    if Train:
                        w2dist = (df_train[jj].values.reshape(-1, 1) - df_combined.iloc[:,
                                                                       (ii + 2) * knn: (ii + 3) * knn].values) / (
                                         df_combined.iloc[:, 0:knn].values ** power)  # use df_train
                    else:
                        w2dist = (df_test[jj].values.reshape(-1, 1) - df_combined.iloc[:,
                                                                      (ii + 2) * knn: (ii + 3) * knn].values) / (
                                         df_combined.iloc[:, 0:knn].values ** power)  # use df_test

                    w2dists_col = [f'w{ii + 2}dist_{i + 1}' for i in range(knn)]
                    w2dists.append(w2dist)
                    w2dists_cols += w2dists_col
                w2dist_df = pd.DataFrame(np.concatenate(w2dists, axis=1), columns=w2dists_cols)  # w2dist_0

        if Train:
            rf_table = pd.concat([df_train.iloc[:, slice(df_train.columns.get_loc(date) + 1)], wdist_df, w2dist_df],
                                 axis=1) if Train else pd.concat([df_test, wdist_df, w2dist_df], axis=1)  # CRMS data
            rf_table.to_csv(os.path.join(Output_dir, "rf_table_train.csv"))
        else:
            rf_table = pd.concat([df_test, wdist_df, w2dist_df], axis=1)  # Grid data in Long,Lat
            rf_table.to_csv(os.path.join(Output_dir, "rf_table_test.csv"))

        return rf_table

    def random_forest_regression(train_features, train_labels, test_features, importance_values, n_estimators=100,
                                 test_labels=None):  # i:iteration
        # Instantiate model with 100 decision trees
        rf = RandomForestRegressor(n_estimators=n_estimators)

        # Train the model on training data
        rf.fit(train_features, train_labels)

        # Use the forest's predict (interpolation) method on the test data
        interpolations = rf.predict(test_features)

        if test_labels is not None:
            # Calculate the absolute errors
            errors = abs(interpolations - test_labels)
            # Print out the mean absolute error (mae)
            print('Mean Absolute Error:', round(np.mean(errors), 2), 'ppt.')

        # Get numerical feature importances
        importances = list(rf.feature_importances_)
        # List of tuples with variable and importance
        feature_list = train_features.columns
        feature_importances = [(feature, round(importance, 2)) for feature, importance in
                               zip(feature_list, importances)]

        # Add importance table value
        numeric = [pair[1] for pair in feature_importances]
        importance_values.append(numeric)

        return importance_values, interpolations

    def get_output_name(CRMSSite, method_str):
        outName = os.path.splitext(os.path.basename(CRMSSite))[
                      0] + f"_{method_str}_n{str(knn)}.csv"

        return outName

    ########################################################################################################################
    ######  Code part 3: Interpolation #####################################################################################
    ########################################################################################################################

    # Get the date style and moving average window
    date_style, MA_window, date_removal = get_date_info(
        Input_file)  # Get the date style and moving average window (MA_window didn not use for the interpolation)
    print('date style: ', date_style, ', MA_window: ', MA_window, ', date caption: ', date_removal)

    ##### Input files ###########################################
    CRMSSite = os.path.join(Inputspace, Input_file)  # CRMS Station data
    maskFile = os.path.join(Inputspace, "Basin_NAD83_Dissolve.shp")  # Manually Dissolved the basins.
    CRMSLL = os.path.join(Inputspace,
                          "CRMS_Long_Lat.csv")  # Downloaded CRMS lat and lon data from https://www.lacoast.gov/crms_viewer/Map/CRMSViewer

    basinFile = os.path.join(Inputspace,
                             "Basin_NAD83.shp")  # CPRA Basin shapefile. Downloaded from https://cims.coastal.la.gov/Viewer/GISDownload.aspx on 2023/3/20. Basin shapefile will use for RF regression model with basin info
    polygon = gpd.read_file(basinFile)

    ##### Output files ###########################################
    outName = os.path.splitext(os.path.basename(CRMSSite))[0] + "_LongLat.csv"
    CRMSSite_LongLat = os.path.join(Output_dir, outName)  # CRMS Station with Lon and Lat

    ##### make a grid file using a boundary shape file
    # read maskfile
    mask_shp = gpd.read_file(maskFile)
    shapes = mask_shp.geometry[0]  # Get the first shape
    xmin, ymin, xmax, ymax = shapes.bounds  # Get the boundary of the shape

    # Projection for GCS
    prj = osr.SpatialReference()
    prj.ImportFromEPSG(int(GCS.split(":")[1]))  # Set projection to GCS

    # Make a grid with a space degree
    space = int(grid_space)  # to avoid truncation error
    scale_factor = int(1000)
    gridx = np.arange(xmin * scale_factor - 1 / 2 * space, xmax * scale_factor + 1 / 2 * space, space)  # 0.005 degree
    gridy = np.arange(ymin * scale_factor - 1 / 2 * space, ymax * scale_factor + 1 / 2 * space, space)
    gridx = gridx / scale_factor
    gridy = gridy / scale_factor

    CRMS_train_list = ['CRMS_Sta', GCS_name_list[0], GCS_name_list[1], 'x', 'y']  # CRMS train data list

    # Grid for interpolation using projected coordinate system
    if Method == 3 or Method == 4:
        scale_factor2 = 100000  # scale factor for KDTree unit: [m to 100km]
        [X, Y] = np.meshgrid(gridx, gridy)
        interp_grids = np.transpose(np.vstack([X.ravel(), Y.ravel()]))
        grid_df = pd.DataFrame(interp_grids, columns=GCS_name_list)

        Grid_GCS_gdf = create_df2gdf(grid_df, GCS_name_list, [], GCS, output_file=None)
        # Convert the coordinates to 'epsg:6344' which corresponds to NAD83(2011) UTM Zone 15N
        Grid_prj_gdf = convert_gcs2coordinates(Grid_GCS_gdf, PRJ)

        interp_grids_prj = np.transpose(
            np.vstack([Grid_prj_gdf["x"].values / scale_factor2, Grid_prj_gdf["y"].values / scale_factor2]))
        # print (interp_grids_prj)

        ####################################################################################################################
        # Add geospatial information to the grid data
        ####################################################################################################################
        if geoinfo == True:

            # Read DEM datasets
            str_variable1 = 'NGOM2'
            z_from_raster = os.path.join(Inputspace,
                                         f"Grid_{str(grid_space)}_NGOM2.csv")  # Manually read the values. Note: NGOM2 is 1 TB product. We need a large memory on Python when we open the file. Also, some offshore points are gone when we use ArcGIS. Interpolated from QGIS3.22.
            z_from_raster_df = create_dataframe(z_from_raster, date_column=None, usecols=None)
            loc = grid_df.columns.get_loc(GCS_name_list[1]) + 1  # after 'Lat' column
            grid_df.insert(loc, str_variable1,
                           z_from_raster_df[str_variable1])  # Insert the DEM column after 'Lat' column
            CRMS_train_list.append(str_variable1)

            Turn_on_both_Flag = False  # Turn on both SPI and Temp if user wants to use both SPI and Temp # default is False

            if Turn_on_both_Flag == True:
                str_variable2 = 'SPI'
                SPI_File = os.path.join(Inputspace, "SPI.csv")
                grid_SPI_index, SPI_prj_gdf, SPI_df = process_nClimGrid_data(str_variable2, SPI_File, GCS,
                                                                             GCS_name_list,
                                                                             PRJ, Grid_prj_gdf, scale_factor2, leafsize,
                                                                             xy_list, str_variable1, grid_df, polygon,
                                                                             Output_dir)

                # Read average temperature datasets
                str_variable3 = 'Temp'
                Temp_File = os.path.join(Inputspace, "Tave.csv")
                grid_Temp_index, Temp_prj_gdf, Temp_df = process_nClimGrid_data(str_variable3, Temp_File, GCS,
                                                                                GCS_name_list, PRJ,
                                                                                Grid_prj_gdf, scale_factor2, leafsize,
                                                                                xy_list,
                                                                                str_variable1, grid_df, polygon,
                                                                                Output_dir)
                str_variable_list = [str_variable2, str_variable3]
                print("check the grid_index", grid_Temp_index)

            else:
                # Read SPI datasets
                if y_variable == 'Salinity':
                    str_variable2 = 'SPI'
                    SPI_File = os.path.join(Inputspace, "SPI.csv")
                    grid_SPI_index, SPI_prj_gdf, SPI_df = process_nClimGrid_data(str_variable2, SPI_File, GCS,
                                                                                 GCS_name_list, PRJ, Grid_prj_gdf,
                                                                                 scale_factor2, leafsize, xy_list,
                                                                                 str_variable1, grid_df, polygon,
                                                                                 Output_dir)
                    str_variable_list = [str_variable2]
                    print("check the grid_index", grid_SPI_index)

                # Read average temperature datasets
                else:
                    str_variable3 = 'Temp'
                    Temp_File = os.path.join(Inputspace, "Tave.csv")
                    grid_Temp_index, Temp_prj_gdf, Temp_df = process_nClimGrid_data(str_variable3, Temp_File, GCS,
                                                                                    GCS_name_list, PRJ,
                                                                                    Grid_prj_gdf, scale_factor2,
                                                                                    leafsize,
                                                                                    xy_list,
                                                                                    str_variable1, grid_df, polygon,
                                                                                    Output_dir)
                    str_variable_list = [str_variable3]
                    print("check the grid_index", grid_Temp_index)

            # Add the variable index to the grid data
            for variable in str_variable_list:
                CRMS_train_list.append(variable + '_index')
            CRMS_train_list.append('Basin')

        print('CRMS_train_list is \t', CRMS_train_list)

    ### Parameters of OrdinaryKriging
    # variogram_model = 'spherical'
    # nlags = 6
    # weight = True
    # verbose = False
    # enable_plotting = False
    # n_closest_points =12
    nodata_value = -99999  # 0xff7fffee#-99999
    datasets_summary = []

    # Read CRMS datasets
    df = pd.read_csv(CRMSSite, delimiter=',')
    df = pd.DataFrame(df)

    # Delete columns where 'num_station' is lower than 200
    threshold1 = 200
    row_index = df.loc[df['num_station'] < threshold1].index.tolist()  # Get row index where 'num_station' < threshold
    df = df.drop(row_index)

    # Move datetime into index  ()
    df.index = pd.to_datetime(df.Date)

    df = df.query('index >= @sdate and index <= @edate')  # compare the data between 2008 - 2022

    ########################################################################################################################
    ### User can change the interpolation step every x hours by providing the tstep hours

    # Set x hours for nearest hour selection (e.g., every 3 hours)
    if date_style == 'Y':
        print(f"Selecting rows at every {tstep} years")
    elif date_style == 'M':
        print(f"Selecting rows at every {tstep} months")
    elif date_style == 'D':
        print(f"Selecting rows at every {tstep} days")
    elif date_style == 'H':
        print(f"Selecting rows at every {tstep} hours")

    # Select rows at every x steps using slicing
    df = df.iloc[::tstep]
    try:
        df.index = df.index.strftime(date_style)

    except Exception as e:
        raise ValueError("The file name must contain one of 'Ydata', 'Mdata', 'Ddata' or 'Hdata'")

    # Drop the columns that were used to create the index
    df.drop(['Date'], axis=1, inplace=True)
    df_transposed = df.T  # Transpose the dataframe

    # Change the name of first Column to "CRMS Sta" # 10 characters is maximum
    df_transposed = df_transposed.reset_index()
    df_transposed.rename(columns={'index': 'CRMS_Sta'}, inplace=True)

    index_to_drop = df_transposed.loc[df_transposed['CRMS_Sta'] == 'num_station'].index[-1]
    datasets = df_transposed.drop(index_to_drop)

    #### Add long and lat on CRMS stations
    df_CRMSLL = pd.read_csv(CRMSLL)

    # Convert df_CRMSLL to a GeoDataFrame
    column_list = ['CRMS Site', 'Longitude', 'Latitude']
    # Add basin information
    df = Spatial_join_basin(df_CRMSLL, polygon, GCS,
                            column_list)  # Join and keep only the columns we need ['CRMS_Sta','Longitude','Latitude', 'BASIN']

    if basin_info == True:
        df = df[df.eval("Basin != 'Pearl' & Basin != 'Other'")]  # Remove 'Pearl' and 'Other' basins from the train data

    # Merge the datasets and df_CRMSLL
    CRMS_point = pd.merge(datasets, df[column_list + ['NGOM2']], how='left', left_on=datasets.columns[0],
                          right_on=df.columns[0])
    CRMS_point.rename(columns={'Longitude': GCS_name_list[0], 'Latitude': GCS_name_list[1]},
                      inplace=True)  # Change column name 'Longitude' to 'Long' and 'Latitude' to 'Lat'
    CRMS_point.dropna(subset=GCS_name_list, inplace=True)
    CRMS_point.drop(['CRMS Site'], axis=1, inplace=True)

    column_reorder = [0, CRMS_point.columns.get_loc(GCS_name_list[0]), CRMS_point.columns.get_loc(GCS_name_list[1]),
                      CRMS_point.columns.get_loc('NGOM2'), CRMS_point.columns.get_loc('Basin')] + list(range(1,
                                                                                                             CRMS_point.shape[
                                                                                                                 1] - 4))  # column name reorder index,# CRMS_point.shape[1] is maximum column number. The order is, Long, Lat, NGOM2 (interpolated z) Basin,CRMS_point.shape[1]-4 is the last column index
    # print(column_reorder)
    CRMS_point = CRMS_point.iloc[:, column_reorder]
    CRMS_point.to_csv(CRMSSite_LongLat, index=False)  # Save merged CSV file

    ########################################################################################################################
    # Create station shape file based on input data

    if "Surface_salinity" in CRMSSite:
        CRMS_stations = os.path.join(Output_dir, "CRMS_stations_Surface_salinity.shp")
    elif "Geoid99_to_Geoid12b_offset" in CRMSSite:
        CRMS_stations = os.path.join(Output_dir, "CRMS_stations_Water_Elevation_to_Datum.shp")
    elif "Water_Elevation_to_Marsh" in CRMSSite:
        CRMS_stations = os.path.join(Output_dir, "CRMS_stations_Water_Elevation_to_Marsh.shp")
    elif "CRMS_Water_Temp" in CRMSSite:
        CRMS_stations = os.path.join(Output_dir, "CRMS_stations_Water_Temp.shp")
        # print(df.index)
    else:
        raise ValueError(
            "The file name must contain one of 'Surface_salinity', 'Water_Elevation_to_Datum', 'Water_Elevation_to_Marsh' or 'CRMS_Water_Temp'")

    slice_index = CRMS_point.columns.get_loc('Basin') + 1  # for iloc, need to add one
    CRMS_gdf = create_df2gdf(CRMS_point.iloc[:, slice(slice_index)], GCS_name_list, [], GCS,
                             output_file=CRMS_stations)  # Create point shape file for train data
    # Convert the coordinates to 'epsg:6344' which corresponds to NAD83(2011) UTM Zone 15N
    CRMS_prj_gdf = convert_gcs2coordinates(CRMS_gdf, PRJ)

    if geoinfo == True:
        if Turn_on_both_Flag == True:
            CRMS_prj_gdf = get_nearest_variable_index(CRMS_prj_gdf, SPI_prj_gdf, str_variable2, scale_factor2,
                                                      leafsize=leafsize, xy_list=xy_list)
            CRMS_prj_gdf.to_csv(os.path.join(Output_dir, "CRMS_SPI_index.csv"), index=True)  # Save merged CSV file
            CRMS_prj_gdf2 = get_nearest_variable_index(CRMS_prj_gdf, Temp_prj_gdf, str_variable3, scale_factor2,
                                                       leafsize=leafsize, xy_list=xy_list)
            CRMS_prj_gdf2.to_csv(os.path.join(Output_dir, "CRMS_Temp_index.csv"), index=True)  # Save merged CSV file
        else:
            if y_variable == 'Salinity':
                CRMS_prj_gdf = get_nearest_variable_index(CRMS_prj_gdf, SPI_prj_gdf, str_variable2, scale_factor2,
                                                          leafsize=leafsize, xy_list=xy_list)
                CRMS_prj_gdf.to_csv(os.path.join(Output_dir, "CRMS_SPI_index.csv"), index=True)  # Save merged CSV file
            else:
                CRMS_prj_gdf2 = get_nearest_variable_index(CRMS_prj_gdf, Temp_prj_gdf, str_variable3, scale_factor2,
                                                           leafsize=leafsize, xy_list=xy_list)
                CRMS_prj_gdf2.to_csv(os.path.join(Output_dir, "CRMS_Temp_index.csv"),
                                     index=True)  # Save merged CSV file

        # CRMS_prj_gdf.to_file(CRMS_stations, driver='ESRI Shapefile') # Overwrite the shapefile for comfirmation.

    ########################################################################################################################
    #### For loop for each datetime ####
    ########################################################################################################################

    # To identify the importance of each variable in the IDW method
    importance_table = pd.DataFrame()  # 'impotance_rf_' + y_variable '.xlsx'
    if Method == 3 or Method == 4:
        importance_values = []  # store importance variable values

    # Get the column index of date
    for i, col in enumerate(CRMS_point.columns):
        if col.startswith(date_removal):
            slice_index = i
            break

    datelist = list(CRMS_point.columns[slice_index:])
    print(datelist)
    nmax = len(datelist)  # Get the number of columns from the date index to the end of the columns

    for i, date in enumerate(datelist):  # test case
        print(date)
        station = pd.concat([CRMS_prj_gdf.loc[:, CRMS_train_list], CRMS_point[date]], axis=1).dropna()
        station.to_csv(os.path.join(Output_dir, "station.csv"), index=True)  # Save merged CSV file
        date_numeric = date.replace(date_removal, '')  # Get the index of the date column

        if geoinfo == True:
            if Turn_on_both_Flag == True:
                station = assign_variable_index2values(station, SPI_df, date_numeric, str_variable1,
                                                       str_variable2)  # Assign the values of the date column to the station DataFrame

                station = assign_variable_index2values(station, Temp_df, date_numeric, str_variable2,
                                                       str_variable3)  # Assign the values of the date column to the station DataFrame

            else:
                if y_variable == 'Salinity':
                    station = assign_variable_index2values(station, SPI_df, date_numeric, str_variable1, str_variable2)
                else:
                    station = assign_variable_index2values(station, Temp_df, date_numeric, str_variable1, str_variable3)
            print("check station columns", station.columns)
        print("dataset: \t\t", date, station.shape)

        # Create point shape file
        CRMS_point_shp = str(Output_dir) + "/CRMS_points_" + date + ".shp"
        drop_list = xy_list
        create_df2gdf(station, [GCS_name_list[0], GCS_name_list[1]], drop_list, GCS,
                      output_file=CRMS_point_shp)  # Create point shape file for train data

        # Spatial interpolation using selected method
        if Method == 1:
            InterpolatedRaster_tif = kriging_interpolation(station, date, gridx, gridy, knn)
            print(InterpolatedRaster_tif)
        elif Method == 2:
            import whitebox

            wbt = whitebox.WhiteboxTools()
            InterpolatedRaster_tif = idwf_interpolocation(CRMS_point, date, gridx, gridy)
            print(InterpolatedRaster_tif)
        elif Method == 3 or Method == 4:

            # We need scipy for this method. from scipy.spatial import cKDTree as KDTree

            # Use a projected system
            xy = station[['x', 'y']].to_numpy() / scale_factor2

            # eps = .1  # approximate nearest, dist <= (1 + eps) * true nearest
            # weights ~ 1 / distance**p

            if Method == 3:
                # Call KDTree python file ()
                z = station.loc[:, date].values
                invdisttree = Invdisttree(xy, z, leafsize=leafsize, stat=1)  # create i
                interpol, weight_factors = invdisttree(interp_grids_prj, nnear=knn, eps=0.0,
                                                       p=power)  # interpolated using nnear numbers
                z_interp = np.reshape(interpol, (len(gridy), len(gridx)))  # original interpol is 1D
                InterpolatedRaster_tif = gdal_tiff(z_interp, "IDW_KNN_process.tif")
                # print(InterpolatedRaster_tif)

                # Add weighted factors into the importance table
                # np.savetxt (os.path.join(Output_dir, "weight_factors.csv"),weight_factors, delimiter=",",fmt='%s') # confirmed the contents

                # Calculate the average of the weighted factors
                if knn == 1:
                    Averaged_importance = [1]
                else:
                    Averaged_importance = [np.mean([sub_arr[i] for sub_arr in weight_factors]) for i in range(knn)]

                importance_values.append(Averaged_importance)

            elif Method == 4:
                dist_tree = KDTree(xy, leafsize=leafsize)  # creates the distance binary tree

                # Reset index to get k nearest stations names for merging
                CRMS_station_id_reset = station.copy().reset_index()

                z_train_list = [date]  # minimum requirement for the RF model

                if geoinfo == True:
                    if Turn_on_both_Flag == True:
                        for variable in [str_variable1, str_variable2, str_variable3]:
                            z_train_list.append(variable)
                    else:
                        if y_variable == 'Salinity':
                            for variable in [str_variable1, str_variable2]:
                                z_train_list.append(variable)
                        else:
                            for variable in [str_variable1, str_variable3]:
                                z_train_list.append(variable)

                print('z_train_list:\t', z_train_list)

                rf_CRMS_train = process_nearest_neighbors_RF(dist_tree, xy, CRMS_station_id_reset, z_train_list, knn,
                                                             power,
                                                             date, Train=True,
                                                             df_test=None)  # z and CRMS_train_drop are always train data. (dist_tree, xy, df_train, z_train_list, knn, power, date, Train=True, num_numeric_variable_list=[], df_test=None):

                # # checked CRMS site. This part will keep even replaced by grid points data (benchmark)
                # xy_test = CRMS_test[['x', 'y']].to_numpy()
                # xy_test = xy_test / scale_factor2
                # z_test = CRMS_test.loc[:, date].values

                # Grid data
                if geoinfo == True:
                    if Turn_on_both_Flag == True:
                        grid_df_dummy2 = grid_Temp_index.copy()  # Assign the values of the date column to the grid DataFrame
                        grid_df2 = assign_variable_index2values(grid_df_dummy2, Temp_df, date_numeric, str_variable1,
                                                                str_variable3)  # Assign the values of the date column to the grid DataFrame
                        grid_df = assign_variable_index2values(grid_df2, SPI_df, date_numeric, str_variable1,
                                                               str_variable2)
                    else:
                        if y_variable == 'Salinity':
                            grid_df_dummy = grid_SPI_index.copy()  # Always use a copied grid data to avoid overwriting
                            grid_df = assign_variable_index2values(grid_df_dummy, SPI_df, date_numeric, str_variable1,
                                                                   str_variable2)  # Assign the values of the date column to the grid DataFrame

                        else:
                            grid_df_dummy = grid_Temp_index.copy()  # Always use a copied grid data to avoid overwriting
                            grid_df = assign_variable_index2values(grid_df_dummy, Temp_df, date_numeric, str_variable1,
                                                                   str_variable3)

                    # grid_df_dummy = grid_df_index.copy()  # Always use a copied grid data to avoid overwriting
                    # grid_df = assign_variable_index2values(grid_df_dummy, SPI_df, date_numeric,str_variable1, str_variable2) # Assign the values of the date column to the grid DataFrame

                    # Keep/Remove the geo_info labels from test features
                    if basin_info == True:
                        remove_columns_grid = []
                    else:
                        remove_columns_grid = ['Basin']
                else:
                    remove_columns_grid = GCS_name_list  # 'SPI' and 'Basin' are not assigned to the grid data'

                rf_Grid_test = process_nearest_neighbors_RF(dist_tree, interp_grids_prj, CRMS_station_id_reset,
                                                            z_train_list,
                                                            knn, power, date, Train=False,
                                                            df_test=grid_df)  # z and CRMS_train_drop are always train data
                # print(rf_Grid_test.columns)

                remove_train_common = ['index', 'CRMS_Sta', date] + xy_list  # Remove common columns from the train data
                remove_train_columns = remove_train_common + remove_columns_grid

                # finalize train and test data
                train_features = rf_CRMS_train.drop(remove_train_columns, axis=1)
                train_labels = rf_CRMS_train.loc[:, date]
                test_features = rf_Grid_test.drop(remove_columns_grid, axis=1)

                # One-hot encoding for basin info
                if 'Basin' in train_features.columns.tolist():
                    train_features = pd.get_dummies(train_features, columns=['Basin'], dtype=int)
                    test_features = pd.get_dummies(test_features, columns=['Basin'],
                                                   dtype=int)  # One-hot encoding for basin info
                    if 'Basin_Pearl' in test_features.columns:
                        columns_to_drop = ['Basin_Pearl',
                                           'Basin_Other']  # Remove 'Pearl' and 'Other' basins from the test data
                    else:
                        columns_to_drop = ['Basin_Other']  # Remove ''Other' basins from the test data
                    test_features = test_features.drop(columns_to_drop, axis=1)

                train_features.to_csv(os.path.join(Output_dir, "train_features.csv"), index=True)
                test_features.to_csv(os.path.join(Output_dir, "test_features.csv"), index=True)

                importance_values, interpol = random_forest_regression(train_features, train_labels, test_features,
                                                                       importance_values,
                                                                       n_estimators=500, test_labels=None)

                # print('importance_table',importance_table)
                # importance_table.to_csv(os.path.join(Output_dir, "importance_table2.csv"), index=True)

                z_interp = np.reshape(interpol, (len(gridy), len(gridx)))  # original interpol is 1D
                InterpolatedRaster_tif = gdal_tiff(z_interp, "RF_KNN_process.tif")
                # print(InterpolatedRaster_tif)

            else:
                pass  # Feature modification
        else:
            pass  # Feature modification

        # Extract raster values at points
        extract_point_values(InterpolatedRaster_tif, CRMS_point_shp)

        gdf = gpd.read_file(CRMS_point_shp)
        gdf = calculate_gdf_error(gdf, date)
        # gdf.to_file(CRMS_point, driver='ESRI Shapefile')

        datasets_summary.append(
            gdf.loc[:, [date, 'interp', 'error']].values.tolist())  # observed value,interpolated value, error
        outName = get_output_name(CRMSSite, method_str)
        merged_csv_file = os.path.join(Output_dir, outName)

        if i == 0:
            shp1 = gpd.read_file(CRMS_stations)
        else:
            shp1 = pd.read_csv(merged_csv_file)
        # shp2 = gpd.read_file(CRMS_point_test)
        shp2 = gdf.copy()
        merged_shp = merge_shapefiles(shp1, shp2, date, i)
        if i != nmax:
            merged_shp.to_csv(merged_csv_file, index=False)  # Save merged CSV file
        else:
            merged_shp.drop('geometry', axis=1).to_csv(merged_csv_file, index=False)  # Save merged CSV file
        # Open the TIFF file and clip using a polygon
        with rasterio.open(InterpolatedRaster_tif) as src:
            raster = src.read(1)
            transform = src.transform
            # Clip the raster to the polygon boundary
            raster, transform = mask(src, mask_shp.geometry, crop=True, all_touched=True, nodata=np.nan)
            Raster_clip = write_raster(src, raster, transform, nodata_value, method_dict, Method, i, date, Output_dir)

        # Create Contour line (optional)
        Contour_name = os.path.join(Output_dir, f"{y_variable}_MA_Contour_{date}_n{knn}_{method_str}.shp")
        # Create the GeoTIFF file
        Contour_file = create_contour_line(Raster_clip, nodata_value, Contour_name, contour_list=None)

    # Create an importance table and add stats for importance of variables
    if Method == 3 or Method == 4:

        importance_values_array = np.array(importance_values)
        importance_values_2d = importance_values_array.reshape(importance_values_array.shape[0], -1)
        importace_columns = ['trial_' + str(ii) for ii in range(nmax)]  # i months
        if Method == 3:
            importance_index = ['wdist_' + str(i) for i in range(1, knn + 1)]
        else:
            importance_index = train_features.columns.tolist()
        importance_table = pd.DataFrame(importance_values_2d.T, columns=importace_columns, index=importance_index)

        # Add statistics to the importance table
        importance_table = add_stats(importance_table, importance_table.columns.get_loc('trial_0'))
        outName = "Impotance" + f"_{method_str}_" + y_variable + ".csv"
        output_dir = os.path.join(Output_dir, outName)
        importance_table.to_csv(output_dir, index=True, header=True)

    ########################################################################################################################
    # Geostatistical Analysis # Error Analysis
    ########################################################################################################################

    df_error = create_error_sheet(merged_csv_file)
    outName = os.path.splitext(os.path.basename(CRMSSite))[0] + f"_{method_str}_n{knn}_Error.csv"
    CRMSSite_error = os.path.join(Output_dir, outName)  # CRMS Station with Lon and Lat
    df_error.to_csv(CRMSSite_error, index=False)  # Save merged shapefile

    # threshold2 = len(np.arange(5, shp1.shape[1], 2))/3
    # row_index = df_error.loc[df_error['num_sample'] < threshold2].index.tolist() # Get row index where 'num_sample' < threshold
    # df_error = df_error.drop(row_index)

    # Make a summary of the datasets in each CRMS station
    df_error = df_error[df_error[
                            'num_sample'] != 0]  # When bootstrap number is small, np.nan affect the results # Drop the rows where 'num_sample' is equal to 0
    gdf_error_RMSE = gpd.GeoDataFrame(
        df_error.loc[:, ['CRMS_Sta', GCS_name_list[0], GCS_name_list[1], 'num_sample', 'MAE', 'RMSE', 'RMSSE']],
        geometry=gpd.points_from_xy(df_error[GCS_name_list[0]], df_error[GCS_name_list[1]], df_error['RMSE'],
                                    crs=GCS))
    # gdf_error_RMSE.to_csv(os.path.join(Output_dir, 'gdf_error_RMSE.csv'), index=False, header=True)
    # print(gdf_error_RMSE.describe())

    outName = os.path.splitext(os.path.basename(CRMSSite))[0] + "_RMSE.shp"
    CRMSData = os.path.join(Output_dir, outName)
    gdf_error_RMSE.to_file(CRMSData, driver='ESRI Shapefile')

    gdf_error_MAE = gpd.GeoDataFrame(
        df_error.loc[:, ['CRMS_Sta', GCS_name_list[0], GCS_name_list[1], 'num_sample', 'MAE', 'RMSE', 'RMSSE']],
        geometry=gpd.points_from_xy(df_error[GCS_name_list[0]], df_error[GCS_name_list[1]], df_error['MAE'],
                                    crs=GCS))
    outName = os.path.splitext(os.path.basename(CRMSSite))[0] + "_MAE.shp"
    CRMSData = os.path.join(Output_dir, outName)
    gdf_error_MAE.to_file(CRMSData, driver='ESRI Shapefile')

    # Make a summary of the datasets of all runs
    outName = os.path.splitext(os.path.basename(CRMSSite))[0] + f"_{method_str}_n{knn}_datasets_summary.csv"
    CRMSSite_datasets_summary = os.path.join(Output_dir, outName)  # all CRMS Stations data

    df = pd.DataFrame(datasets_summary)
    # df.to_csv(os.path.join(Output_dir, 'datasets_summary_check.csv'), index=False, header=True)

    # For plot
    result = []

    for row in df.values:
        for i in row:
            # print(i)
            i_str = str(i).replace('[', '').replace(']', '')
            contents = i_str.split(",")
            try:
                if contents[0] == 'nan':
                    pass
                elif contents[0] == 'None':
                    pass
                else:
                    result.append(contents)
            except TypeError:
                print('something wrong')

    # Create a DataFrame with three columns
    df2 = pd.DataFrame(result, columns=['obs', 'interpolation', 'error'])
    df2.to_csv(CRMSSite_datasets_summary, index=False, header=True)

    # Add calculate statistics
    df3 = calculate_statistics(CRMSSite_datasets_summary)
    df3.to_csv(CRMSSite_datasets_summary, index=False, header=True)

    # Calculate the elapsed time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print("Done Step 3")
    print("Time to Compute: \t\t\t", elapsed_time, " seconds")
    print("Job Finished ʕ •ᴥ•ʔ")

    pass
