#!/usr/bin/env python
# coding: utf-8
# Provide CRMS2Interpolation functions
# Developed by the Center for Computation & Technology and Center for Coastal Ecosystem Design Studio at Louisiana State University (LSU).
# Developer: Jin Ikeda and Christopher E. Kees
# Last modified Feb 23, 2024

import numpy as np
import pandas as pd
import geopandas as gpd
import os, sys, zipfile, shutil
from datetime import datetime
import time
import glob
from pathlib import Path
import rasterio
from osgeo import gdal, ogr, osr
from osgeo.gdalconst import *
gdal.AllRegister()  # Register all of drivers
ogr.UseExceptions()  # Use exceptions for errors
import matplotlib.pyplot as plt
import seaborn as sns

########################################################################################################################
# y_variable and date_style for data processing
########################################################################################################################
def download_CRMS(url, zip_file, csv_file, folder_path):
    # Remove the existing CSV file if it exists
    csv_file = os.path.join(folder_path, csv_file)
    if os.path.exists(csv_file):
        os.remove(csv_file)

    # Use wget to download the file
    try:
        os.system(f"wget {url}")
        print("Downloaded an original dataset on linux")

        import requests
        print ('wget command is not recognized on your system (e.g, Windows). So, use another method..')
        response = requests.get(url)
        with open(csv_file, 'wb') as f:
            f.write(response.content)
        print("Downloaded an original dataset on windows")

    except:
        print ("failed to download the file")

    # Use zipfile to extract the zip file
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith('.csv') or file.endswith('.pdf'):
                zip_ref.extract(file, folder_path)

    # Remove the zip file
    os.remove(zip_file)

def get_CRMS_file(Data):
    data_to_filename = {
        1: "CRMS_Surface_salinity.csv",
        2: "CRMS_Geoid99_to_Geoid12a_offsets.csv",
        3: "CRMS_Water_Elevation_to_Marsh.csv",
        4: "CRMS_Water_Temp.csv",
        5: "Something_wrong",
        6: "CRMS_Water_Elevation_to_Marsh.csv", # "CRMS_Water_Elevation_to_Marsh_2006_2024_wd.csv", # hidden file for hydrographic data
        7: "CRMS_Water_Elevation_to_Marsh.csv",# "CRMS_Water_Elevation_to_Marsh_2006_2024_wdepth.csv" # hidden file for inundation data
    }

    file = data_to_filename.get(Data,
                                     "Something_wrong")  # default to "Something_wrong" if Data is not in the dictionary
    if file == "Something_wrong":
        raise ValueError(
            "The file name must be one of 'CRMS_Surface_salinity', 'CRMS_Geoid99_to_Geoid12a_offsets', 'CRMS_Water_Elevation_to_Marsh' or 'CRMS_Water_Temp'")

    return file


def get_y_variable(Input_file):
    if 'salinity' in Input_file or 'Salinity' in Input_file:
        y_variable = 'Salinity'
    elif 'Geoid99_to_Geoid12a' in Input_file or 'WL' in Input_file:
        y_variable = 'WL'
    elif 'Water_Elevation_to_Marsh' in Input_file and 'wd.' in Input_file: # Need to add comma
        y_variable = "W_HP"
    elif 'Water_Elevation_to_Marsh' in Input_file and 'wdepth' in Input_file:
        y_variable = "W_depth"
    elif 'Water_Elevation_to_Marsh' in Input_file:
        y_variable = "WL2M"
    elif 'Temp' in Input_file:
        y_variable= "Temp"
    else:
        print('variable will be added \t', "TBD")  # Feature modification
        y_variable = None
    return y_variable

def get_date_info(Input_file):
    # H: hourly, D: dayly, M: monthly, Y: yearly
    date_formats = {"Ydata": 'Y%Y',"Mdata": 'M%Y_%m',"Ddata": 'D%Y_%m%d',"Hdata": 'H%Y_%m%d_%H'}
    MA_formats = {"Mdata": 12, "Ddata": 365,"Hdata": 8766} # Set a moving window range for yearly analysis
    date_formats_removal = {"Ydata": 'Y',"Mdata": 'M',"Ddata": 'D',"Hdata": 'H'}

    for key, value in date_formats.items():
        if key in Input_file:
            date_style = value
            break
    else:
        raise ValueError("The file name must contain one of 'Ydata', 'Mdata', 'Ddata' or 'Hdata'")

    for key, value in MA_formats.items():
        if key in Input_file:
            MA_window = value
            break
    else:
        raise ValueError("The file name must contain one of 'Mdata', 'Ddata' or 'Hdata'")

    # Drop the columns that were used to create the index
    for key, value in date_formats_removal.items():
        if key in Input_file:
            date_removal = value
            break

    return date_style, MA_window, date_removal

def get_file_lists(Workspace,folder_pattern,file_pattern , methods, knn_values, y_variable):
    Input_file_list = []
    for method, knn in zip(methods, knn_values):
        # Create the folder pattern
        if folder_pattern is not None:
            formatted_folder_pattern = folder_pattern.format(y_variable, knn, method)
            print(formatted_folder_pattern)
            file_path = os.path.join(Workspace, formatted_folder_pattern, file_pattern)
        else:
            file_path = os.path.join(Workspace, file_pattern)

        # Get a list of all matching CSV files
        file = glob.glob(file_path)
        print(file)

        # Append the first file to the list
        Input_file_list.append(file)

    # Flatten the list from 2D to 1D
    Input_file_list = [item for sublist in Input_file_list for item in sublist]

    return Input_file_list

def get_folder_lists(Workspace, folder_pattern, methods, knn_values, y_variable):
    folder_lists = {}
    for method, knn in zip(methods, knn_values):
        folder_list = get_matching_folders(Workspace, folder_pattern, method, knn, y_variable)
        folder_lists[method] = folder_list
    return folder_lists

# subfunction of get_folder_lists
def get_matching_folders(Workspace, folder_pattern, method, knn, y_variable):
    # example: folder_pattern = f"Output_{y_variable}_n{knn}_{input_method}"

    # Format the folder and file patterns with the actual values
    formatted_folder_pattern = folder_pattern.format(y_variable, knn, method)
    # Create the file path
    folder_path = os.path.join(Workspace, formatted_folder_pattern)
    # Get a list of all matching folders
    folders = glob.glob(folder_path)
    return folders

def save_to_excel(outName, data_dict):
    with pd.ExcelWriter(outName) as writer:
        for sheet_name, df in data_dict.items():
            df.to_excel(writer, sheet_name=sheet_name)

########################################################################################################################
# Pandas DataFrame and GeoDataFrame functions
########################################################################################################################
def create_dataframe(file_name ,date_column='Date', sheet_name=None, usecols = None):

    # Use the appropriate pandas function to read the file
    if '.xlsx' in file_name:
        try:
            if sheet_name is None:
                sheet_name = 0 # Default to the first sheet if no sheet name is provided
            df = pd.read_excel(file_name,sheet_name=sheet_name,usecols=usecols)
        except UnicodeDecodeError:
            print('encoding error')            # CRMS = pd.read_csv(file, encoding='utf-8')
    elif '.csv' in file_name:
        try:
            df = pd.read_csv(file_name,usecols=usecols)
        except UnicodeDecodeError:
            print('encoding error')            # CRMS = pd.read_csv(file, encoding='utf-8')
    else:
        raise ValueError("Unsupported file type. The file must be a .xlsx or .csv file.")

    # Convert the DataFrame to the correct format
    df = pd.DataFrame(df)

    # print(df.head(5))
    # print(df.shape, df.dtypes)  # Check data size and type

    # Set the index to the 'Date' column and drop the original 'Date' column
    if 'Date' in df.columns:
        try:
            df.index = pd.to_datetime(df[date_column])
            df.drop([date_column], axis=1, inplace=True)
        except:
            print('Date column not found')
    # Future revision may consider filtering of datasets

    return df

def df_move_row_after(df, row_to_move, target_row):
    # Store the row to move
    df_stored_data = df.loc[row_to_move].copy()

    # Drop the row to move
    df.drop(row_to_move, inplace=True)

    # Get the index of the row to place after
    index = df.index.get_loc(target_row)

    # Split the DataFrame into two parts: before and after the row to place after
    df1 = df.iloc[:index+1]
    df2 = df.iloc[index+1:]

    # Append the stored row to the first part
    df1.loc[row_to_move] = df_stored_data

    # Concatenate the two parts to get the final DataFrame
    df = pd.concat([df1, df2])

    return df


def rename_and_set_index(df, rename_str=None):
    if rename_str is not None:
        df.rename(columns={'Unnamed: 0': rename_str}, inplace=True)
        df.set_index(rename_str, inplace=True)
    else:
        df.rename(columns={'Unnamed: 0': 'variables'}, inplace=True)
        df.set_index('variables', inplace=True)
    return df


# make a dataframe of TFZ and TOZ areas
def Area_process_dataframe(file_name, date_removal, sheet_name=None):
    # Create the DataFrame
    df = create_dataframe(file_name, date_column='dummy', sheet_name=sheet_name)
    #print(df.head())
    df.columns = df.columns.str.replace(date_removal, '')  # Remove the date_removal from the column names
    rename_and_set_index(df)  # Rename the Undamed:0 and set the index
    df.index.rename('Basin', inplace=True)  # Rename the index
    T_df = df.T  # Transpose the DataFrame
    T_df.index = pd.to_datetime(T_df.index, format='%Y_%m')  # Convert the index to datetime

    return T_df


# exclude_stats
def drop_transpose_stats(df):
    df_processed = df.copy()
    df_processed.drop(['const'],errors='ignore', axis=0, inplace=True) # drop index 'const'
    df_processed.drop(['Mean','STD','Q1','Q2','Q3'], axis=1, inplace=True) # drop Mean,STD,Q1,Q2,Q3
    df_T = df_processed.T.reset_index(drop=True)
    df_T.index.rename('trial', inplace='True')
    return df_T

def create_df2gdf(df, xy_list, drop_list, crs, output_file=None):
    gdf = gpd.GeoDataFrame(df.drop(drop_list, axis=1), geometry=gpd.points_from_xy(df[xy_list[0]], df[xy_list[1]], crs=crs))

    if output_file != None:
        gdf.to_file(output_file, driver='ESRI Shapefile')

    return gdf

def convert_gcs2coordinates(gdf, PRJ):
    gdf_proj = gdf.to_crs(PRJ)
    gdf_proj["x"] = gdf_proj.geometry.apply(lambda point: point.x)
    gdf_proj["y"] = gdf_proj.geometry.apply(lambda point: point.y)
    return gdf_proj

def Spatial_join_basin(df, polygon,GCS,column_list):
    # Perform the spatial join
    if 'BASIN' in polygon.columns:
        polygon.rename(columns={'BASIN': 'Basin'}, inplace=True)
        # Check if df has 'Longitude' and 'Latitude' columns
    if 'Longitude' in df.columns:
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude, crs=GCS))
    elif 'Long' in df.columns:
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Long, df.Lat, crs=GCS))
    else:
        raise ValueError("DataFrame does not have longitude and latitude columns")

    # Create a copy of dataframe and set 'Basin' to 'Other'
    df_basin = gdf.copy()
    df_basin['Basin'] = 'Other'

    # Perform the spatial join
    gdf = gpd.sjoin(gdf, polygon, how='left', predicate='within')

    # Find the indices where 'Basin' is not NaN in gdf
    row_index = gdf[gdf['Basin'].notna()].index

    # Replace the 'Basin' values in gdf_basin with the corresponding values in gdf for the indices found
    df_basin.loc[row_index, 'Basin'] = gdf.loc[row_index, 'Basin']

    # add basin info to the column list
    if 'Basin' not in column_list:
        column_list.append('Basin')

    return df_basin


########################################################################################################################
# Raster and Shapefile functions
########################################################################################################################
def read_raster(Rasterdata, GA_stype=GA_ReadOnly): #GA_ReadOnly (0) or GA_Update(1)
    Read_raster = gdal.Open(Rasterdata, GA_stype)
    if Rasterdata is None:
        sys.exit('\tCould not open {0}.'.format(Rasterdata))

    # transform[0] # Origin x coordinate
    # transform[1] # Pixel width
    # transform[2] # x pixel rotation (0° if image is north up)
    # transform[3] # Origin y coordinate
    # transform[4] # y pixel rotation (0° if image is north up)
    # transform[5] # Pixel height (negative)

    # Coordinate system
    prj = Read_raster.GetProjection()  # Read projection
    print("\tProjection:", prj)

    # Get raster size and band
    rows = Read_raster.RasterYSize  # number of rows
    cols = Read_raster.RasterXSize  # number of columns
    bandnum = Read_raster.RasterCount  # band number
    print("\trows=", rows, "\tcols=", cols)
    # print("band=", bandnum)

    # Get georeference info
    transform = Read_raster.GetGeoTransform()
    xOrigin = transform[0]  # Upperleft x
    yOrigin = transform[3]  # Upperleft y
    pixelWidth = transform[1]  # cell size x
    pixelHeight = transform[5]  # cell size y (value is negative)
    print("\txOrigin=", xOrigin, "deg", "yOrigin=", yOrigin, "deg")
    print("\tpixelWidth=", pixelWidth, "deg", "pixelHeight=", -pixelHeight, "deg")  # pixelHeight is always negative

    return prj, rows, cols, transform,bandnum

def read_raster_values(Rasterdata, GA_stype= GA_ReadOnly): #GA_ReadOnly (0) or GA_Update(1)
    Read_raster = gdal.Open(Rasterdata, GA_stype)
    if Rasterdata is None:
        sys.exit('\tCould not open {0}.'.format(Rasterdata))

    # Read the raster band
    band = Read_raster.GetRasterBand(1)
    # Data type of the values
    #print('\tdata type is', gdal.GetDataTypeName(band.DataType))  # Each raster file has a different data type
    # Get band value info
    RV = Read_raster.GetRasterBand(1).ReadAsArray()  # raster values in the band
    RV_1D = RV.flatten() # Convert to 1D array
    RV_1D = RV_1D.reshape(1, -1) # Reshape to have shape (1, xxx)

    return RV_1D, RV

def create_gdal_tiff(z_interp, transform, prj,bandnum, nodata_value, Raster_name):

    # Make a tiff file
    gtiff_driver = gdal.GetDriverByName('GTiff')  # Use GeoTIFF driver

    out_ds = gtiff_driver.Create(Raster_name, z_interp.shape[1], z_interp.shape[0], 1,
                                 gdal.GDT_Float64)  # Create a output file

    out_ds.SetProjection(prj)  # Set the projection directly from the prj string

    out_ds.SetGeoTransform(transform)
    out_band = out_ds.GetRasterBand(bandnum)
    out_band.WriteArray(z_interp)
    #out_band.WriteArray(np.flipud(z_interp))  # This time we read gdal data so, don't need to use flipud: bottom row <-> top row due to origin of raster file
    out_band = out_ds.GetRasterBand(bandnum).SetNoDataValue(nodata_value)  # Exclude nodata value
    out_band = out_ds.GetRasterBand(bandnum).ComputeStatistics(True)  # Calculate statistics for Raster pyramids (Pyramids can speed up the display of raster data)
    print('maximum value of interpolated data is', np.max(z_interp))

    del out_ds
    return Raster_name

def write_raster(src, raster, transform, nodata_value, method_dict,method, i, date, Outputspace):
    out_meta = src.meta.copy()
    out_meta.update({
        "height": raster.shape[1],
        "width": raster.shape[2],
        "transform": transform,
        "nodata": nodata_value
    })
    method_str = method_dict.get(method, "TBD")
    Raster_clip = f"{Outputspace}/{method_str}_train_{i}_{date}.tif"
    with rasterio.open(Raster_clip, "w", **out_meta) as dest:
        dest.write(raster)
    return Raster_clip


def rasterize_shapefile_to_tiff(shapeFile, base_tifFile, Rasterize_file, layer_Attribute, dtype, nodata_value,
                                stats_flag=False):
    # Load the shapefile and change the BASIN_ID values
    shapefile = gpd.read_file(shapeFile)
    # Define the new order of BASIN_ID
    sort_basin = ['PO', 'BS', 'MR', 'BA', 'TE', 'AT', 'TV', 'ME', 'CS'] # for time-series data
    sort_basin = ['PO', 'BS', 'MR', 'BA', 'TE', 'AT', 'TV', 'ME', 'CS', 'Pearl'] # for mapping data
    # Create a dictionary that maps the original BASIN values to the new BASIN_ID values
    basin_id_dict = {basin: i + 1 for i, basin in enumerate(sort_basin)}

    shapefile['BASIN_ID'] = shapefile['BASIN'].map(basin_id_dict) # Replace the BASIN_ID values in the shapefile
    shapeFile_out = shapeFile.replace('.shp', '_out.shp')
    shapefile.to_file(shapeFile_out)     # Save the modified shapefile

    # ReOpen the shapefile
    shapefile = ogr.Open(shapeFile_out)
    layer = shapefile.GetLayer()

    # Get the bounds of the shapefile
    prj, rows, cols, transform, bandnum = read_raster(base_tifFile, GA_ReadOnly)

    out_drv = gdal.GetDriverByName('GTiff')
    out_ds = out_drv.Create(Rasterize_file, cols, rows, bandnum, dtype)  # dtype e.g., gdal.GDT_Float64

    # Set the geotransform and projection of the output raster
    out_ds.SetGeoTransform(transform)

    # Set the projection from the input shapefile
    out_ds.SetProjection(prj)

    # Create a new raster band
    band = out_ds.GetRasterBand(1)
    band.Fill(nodata_value)  # Temporary create a raster file filled with nodata value

    # Rasterize the input layer to the output raster
    gdal.RasterizeLayer(out_ds, [1], layer, options=["ATTRIBUTE=" + layer_Attribute])
    band.SetNoDataValue(nodata_value)
    stats = band.ComputeStatistics(0)
    min_val, max_val, mean_val, std_dev_val = stats
    if stats_flag:
        print(
            f'Made a raster file. Statistics:\n\tMinimum: {min_val}, Maximum: {max_val}, Mean: {mean_val}, Standard Deviation: {std_dev_val}')
    else:
        print('Made a raster file')

    shapefile = None
    raster = None

    return basin_id_dict


def reproject_tiff(src_filename, dst_filename, new_EPSG_num):
    # Open the source file
    src_ds = gdal.Open(src_filename)
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(src_ds.GetProjection())

    # Create the destination spatial reference
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(new_EPSG_num)

    # Reproject the raster
    gdal.Warp(dst_filename, src_ds, dstSRS=dst_srs.ExportToWkt())

    # Close the files
    src_ds = None


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


def create_contour_line(raster_file, nodata_value, Contour_name, contour_list=None):
    # Open the raster file
    raster = gdal.Open(raster_file)

    # Get the band and projection
    band = raster.GetRasterBand(1)
    proj = osr.SpatialReference()
    proj.ImportFromWkt(raster.GetProjection())

    # Read the raster data
    rasterArray = band.ReadAsArray()
    rasterNan = nodata_value
    rasterMax = np.nanmax(rasterArray)
    rasterMin = rasterArray[rasterArray != rasterNan].min()

    print("Max raster value is",rasterMax, "[unit]")

    # Create the output shapefile
    contourDs = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource(Contour_name)

    if contourDs is None:
        raise ValueError("Creation of output file failed")

    # Create a layer in the shapefile
    contour_layer = contourDs.CreateLayer("contours", proj)

    # Define the fields in the shapefile
    fieldDef = ogr.FieldDefn("ID", ogr.OFTInteger)
    contour_layer.CreateField(fieldDef)
    fieldDef = ogr.FieldDefn("level", ogr.OFTReal)
    contour_layer.CreateField(fieldDef)

    # Generate the contour lines
    if contour_list is not None:
        if contour_list == 'auto':
            contour_list = list(np.arange(0, np.ceil(rasterMax), 1))
            contour_list.append(
                0.5)  # Add the 0.5 contour line for the coastal oligohaline conditions are defined as salt concentrations between 0.5 and 5 ppt
            contour_list.sort()
        else:
            contour_list = contour_list

        gdal.ContourGenerate(band, 0, 0, contour_list, 1, nodata_value, contour_layer, 0, 1)
        # ContourGenerate(Band srcBand, double contourInterval, double contourBase, int fixedLevelCount, int useNoData, double noDataValue,
        #                Layer dstLayer, int idField, int elevField
        # contour_list = [2,4,10]

    # Destroy the shapefile data source
    contourDs.Destroy()
    # print('Contour done')
    return Contour_name


########################################################################################################################
# Statistical analysis functions
########################################################################################################################
def add_stats(df, first_col): # Caution excel output includes 'xxx. " ' " cannot be treated as numeric and differs from this output.
    # first col=0 for statistical analyssis
    quantiles = df.iloc[:, first_col:].quantile([0.25, 0.5, 0.75], axis=1, numeric_only=True, interpolation='linear')
    df['Mean'] = df.iloc[:, first_col:].mean(axis=1, skipna=True)
    df['STD'] = df.iloc[:, first_col:].std(axis=1, skipna=True)
    df['Q1'] = quantiles.loc[0.25]
    df['Q2'] = quantiles.loc[0.5]
    df['Q3'] = quantiles.loc[0.75]

    return df

def calculate_gdf_error(gdf, date):
    gdf['error'] = gdf[date] - gdf['interp']  # observed value - interpolated value
    return gdf

def merge_shapefiles(shp1, shp2,date, i):
    merged_shp = shp1.merge(shp2.loc[:,[date, 'error', shp2.columns[0]]], on=shp2.columns[0],
                            how='left')
    target_index= merged_shp.columns.get_loc('geometry') # Use the geometry column index to get the index of the columns: date and 'error'
    if target_index+1 + 2 * i < len(merged_shp.columns) and target_index+2 + 2 * i < len(merged_shp.columns):
        merged_shp.rename(columns={merged_shp.columns[target_index+1 + 2 * i]: 'test_' + str(i),
                                                merged_shp.columns[target_index+2 + 2 * i]: 'test_e_' + str(i)}, inplace=True)
    else:
        print("The number of columns in the shapefile is less than expected. Check the output")
    return merged_shp

def create_error_sheet(csv_file):
    data = pd.read_csv(csv_file)
    error_list = [0, 1, 2, 3] + list(np.arange(data.columns.get_loc('test_e_0'), data.shape[1], 2)) # CRMS_Sta, Long, Lat, NGOM2
    df_error = data.iloc[:, error_list]
    data_frame = df_error.iloc[:, df_error.columns.get_loc('test_e_0'):]

    pd.set_option('mode.chained_assignment', None)
    std_column = data_frame.std(axis=1, skipna=True, ddof=0)
    SE_column = data_frame.sem(axis=1, skipna=True, ddof=0)
    MSE_column = data_frame.div(std_column, axis=0).mean(axis=1, skipna=True)
    RMSE_column = (data_frame ** 2).mean(axis=1, skipna=True) ** 0.5
    RMSSE_column = ((data_frame.div(std_column, axis=0)) ** 2).mean(axis=1, skipna=True) ** 0.5
    MAE_column = abs(data_frame).mean(axis=1, skipna=True)

    SE = SE_column.mean()
    MSE = MSE_column.mean()
    RMSE = RMSE_column.mean()
    RMSSE = RMSSE_column.mean()
    MAE = MAE_column.mean()

    print('SE is \t', SE,'MAE is \t', MAE, 'RMSE is \t', RMSE)

    df_error['num_sample'] = data_frame.count(axis=1)
    df_error['RMSE'] = RMSE_column
    df_error['mean_e'] = data_frame.mean(axis=1, skipna=True)
    df_error['std_e'] = std_column
    df_error['MAE'] = MAE_column

    df_error.loc[df_error.eval("num_sample == 1"), ['SE', 'MSE', 'RMSSE']] = 0
    df_error.loc[df_error.eval("num_sample > 1"), 'SE'] = SE_column[df_error.eval("num_sample > 1")]
    df_error.loc[df_error.eval("num_sample > 1"), 'MSE'] = MSE_column[df_error.eval("num_sample > 1")]
    df_error.loc[df_error.eval("num_sample > 1"), 'RMSSE'] = RMSSE_column[df_error.eval("num_sample > 1")]

    return df_error

def calculate_statistics(csv_file): # Calculate statistics using observation and interpolation
    df = pd.read_csv(csv_file)
    refdata = df.iloc[:, 0]  # observation
    model = df.iloc[:, 1]  # interpolation
    data_frame=df.iloc[:, 2] # difference between observation and interpolation
    MEAN =data_frame.mean(axis=0, skipna=True)  # Mean
    STD = data_frame.std(axis=0, skipna=True, ddof=0)  # Population standard deviation
    SE = data_frame.sem(axis=0, skipna=True, ddof=0)  # Mean Standard Error = std/(n**0.5)
    MSE = data_frame.abs().div(STD).mean(axis=0, skipna=True)  # Mean Standardized Error = (error/std)/n
    RMSE = (data_frame ** 2).mean(axis=0, skipna=True) ** 0.5  # Root square mean error
    RMSSE = ((data_frame.div(STD)) ** 2).mean(axis=0,skipna=True) ** 0.5  # Root square mean error #Root Mean Square Standardized Error

    S = data_frame.dropna().shape[0]  # Number of data
    MAE = np.mean(abs(data_frame))
    se = (data_frame ** 2)
    mo = np.sqrt(np.nanmean((refdata) ** 2))  # root mean square of reference data
    SI = STD / mo  # Scatter index
    O = np.nanmean(np.nanmean(model))
    sO = model - O
    P = np.nanmean(np.nanmean(refdata))
    sP = refdata - P
    IA = 1 - np.sum(se) / (np.sum(sO ** 2) + np.sum(sP ** 2) + S * (O - P) ** 2 + 2 * abs(np.sum(sO * sP)))

    df.loc[0,['num_sample']] = S
    df.loc[0,['SE']] = SE
    df.loc[0,['MSE']] = MSE
    df.loc[0,['RMSE']] = RMSE
    df.loc[0,['RMSSE']] = RMSSE
    df.loc[0,['mean_e']] = MEAN
    df.loc[0,['std_e']] = STD
    df.loc[0,['MAE']] = MAE
    df.loc[0,['IA']] = IA

    return df


########################################################################################################################
# Plotting functions
########################################################################################################################
def normalize_rgb(rgb_values):
    return [value/255 for value in rgb_values]

# Define the custom color palette
def create_color_palette():
    def normalize_rgb(rgb_values):
        # Normalize RGB values to the [0, 1] range, which is the format matplotlib accepts.
        return [value/255 for value in rgb_values]

    color_palette_methods = []

    color_palette_methods.append(normalize_rgb([181, 124, 0])) # Strong orange Krige
    color_palette_methods.append(normalize_rgb([223, 115, 255])) # red # IDW
    color_palette_methods.append(normalize_rgb([0, 128, 0]))  # green RF
    color_palette_methods.append(normalize_rgb([30,144,255])) # blue # RF_geo

    return color_palette_methods

def set_color_entries():

    # Create a color table for the classified raster first before writing the array
    colors = gdal.ColorTable()
    # Set the color entries for the classes

    # Set the color for the 0 class as transparent
    colors.SetColorEntry(0, (200, 200, 200, 0))  # Transparent color (0, 0, 0) with alpha channel 0

    # Set colors for other classes
    colors.SetColorEntry(1, (0, 0, 255))  # Color for class 1
    colors.SetColorEntry(2, (0, 255, 128))  # Color for class 2
    colors.SetColorEntry(3, (255, 127, 80))  # Color for class 3
    colors.SetColorEntry(4, (194, 140, 124))  # Color for class 4
    colors.SetColorEntry(5, (214, 193, 156))  # Color for class 5
    colors.SetColorEntry(6, (112, 153, 89))  # Color for class 6

    return colors

def create_raster_attribute(class_values):
    # Create a raster attribute table
    rat = gdal.RasterAttributeTable()

    # Create columns
    rat.CreateColumn('Class_ID', gdal.GFT_Integer, gdal.GFU_Name)
    rat.CreateColumn('Count', gdal.GFT_Integer, gdal.GFU_PixelCount)
    rat.CreateColumn('Area [km2]', gdal.GFT_Real, gdal.GFU_Generic)
    rat.CreateColumn('Range', gdal.GFT_String, gdal.GFU_Generic)

    # Set row count
    rat.SetRowCount(len(class_values))

    return rat


# Define a dictionary for the boundaries
boundaries_dict = {
    'Salinity': [0, 0.5, 5.0, 1000],
    'WL': [-0.2, 0, 0.2, 0.4, 0.6, 1000]
    # Add more y_variables and their boundaries here
}

# Define a function to get the boundaries
def get_boundaries(y_variable):
    return boundaries_dict.get(y_variable, None)

def color_rgb_salinity():
    return [normalize_rgb([0, 0, 255]), normalize_rgb([0, 255, 128]), normalize_rgb([255, 127, 80])]

def color_rgb_WL():
    return [normalize_rgb([0, 0, 255]), normalize_rgb([0, 255, 128]), normalize_rgb([255, 127, 80]),
            normalize_rgb([255, 50, 255]), normalize_rgb([225, 0, 0])]










