#!/usr/bin/env python
# coding: utf-8
# CRMS_Continuous_Hydrographic2subsets
# Developed by the Center for Computation & Technology and Center for Coastal Ecosystem Design Studio at Louisiana State University (LSU).
# Developer: Jin Ikeda, Shu Gao, and Christopher E. Kees
# Last modified Dec 4, 2024

from src.CRMS_general_functions import *


@click.command()
def continuous_subcommand():
    """Handle continuous hydrographic data processing."""

    ### Step 1 #############################################################################################################
    print("Step 1: Auto retrieve the original datasets")
    ########################################################################################################################
    ### 1.1 Import modules ###
    # Import the general interpolation functions from the CRMS_general_functions.py file (some interpolation functions are keep inside of this code)

    start_time = time.time()

    # The target working directory
    # Workspace = "/Users/xxx/yyy/CRMS2Map/" # Local
    Workspace = (
        os.getcwd()
    )  # Get the current working directory (same with the setup.py)

    # Print the current working directory
    print("Current working directory: {0}".format(os.getcwd()))

    Inputspace = os.path.join(Workspace, "Input")  # Make Input folder
    Processspace = os.path.join(Workspace, "Process")  # Make Input folder

    try:
        os.makedirs(Inputspace, exist_ok=True)
        os.makedirs(Processspace, exist_ok=True)

    except Exception as e:
        print(f"An error occurred while creating directories: {e}")

    start_time = time.time()

    file_suffix = ".csv"
    file_suffix_zip = ".zip"
    file_name = "CRMS_Continuous_Hydrographic"

    ########################################################################################################################
    print("Download an original datasets")
    ########################################################################################################################
    # Define the name of the zip file and the csv file
    zip_file = file_name + file_suffix_zip
    csv_file = file_name + file_suffix
    print(csv_file)

    ### 1.2 Automatically download the file
    ########################################################################################################################
    # User can also manually download and upload the Continuous datasets from https://cims.coastal.la.gov/FullTableExports.aspx
    ########################################################################################################################

    url = (
            "https://cims.coastal.la.gov/RequestedDownloads/ZippedFiles/"
            + file_name
            + file_suffix_zip
    )
    download_CRMS(url, zip_file, csv_file, Inputspace)  # Download the file

    print("Downloaded an original dataset")

    # Manually downloaded the offset conversion file
    offsets_name = "GEOID99_TO_GEOID12B"

    file = file_name + file_suffix
    offsets_file = offsets_name + file_suffix

    import gc

    # Read only the CSV header first to map post-drop column indices to original column names
    _enc = "iso-8859-1"
    try:
        _hdr = pd.read_csv(os.path.join(Inputspace, file), nrows=0, encoding=_enc)
    except UnicodeDecodeError:
        _enc = "utf-8"
        _hdr = pd.read_csv(os.path.join(Inputspace, file), nrows=0, encoding=_enc)
    _all_cols = _hdr.columns.tolist()
    del _hdr

    _date_col = "Date (mm/dd/yyyy)"
    _time_col = "Time (hh:mm:ss)"
    _tz_col = "Time Zone"
    _drop_cols = {_date_col, _time_col, _tz_col}
    # Post-drop column indices [0, 3, 7, 11, 13, 16, 41, 42] map to original column names
    _remaining = [c for c in _all_cols if c not in _drop_cols]
    _needed_indices = [0, 3, 7, 11, 13, 16, 41, 42]
    _data_cols = [_remaining[i] for i in _needed_indices]
    _usecols = [_date_col, _time_col, _tz_col] + _data_cols

    offsets = pd.read_csv(os.path.join(Inputspace, offsets_file), encoding=_enc)

    # Read the large CSV in chunks using only needed columns and filtering early to save memory
    print("Reading large dataset in chunks (this may take a while)...")
    _chunksize = 500_000
    _chunks = []
    for _chunk in pd.read_csv(
        os.path.join(Inputspace, file),
        encoding=_enc,
        usecols=_usecols,
        chunksize=_chunksize,
    ):
        # Filter for surface (-H) stations and exclude surrogate (-H0X) early
        _chunk = _chunk[
            _chunk["Station ID"].str.contains("-H")
            & ~_chunk["Station ID"].str.contains("-H0X")
        ]
        if len(_chunk) == 0:
            continue
        _dt = _chunk[_date_col] + " " + _chunk[_time_col]
        _chunk.index = pd.to_datetime(_dt, format="%m/%d/%Y %H:%M:%S")
        _chunk.index = _chunk.index.floor("min")
        _chunk.index.name = "Date"
        _chunk.drop([_date_col, _time_col, _tz_col], axis=1, inplace=True)
        # Downcast float64 to float32 to halve numeric memory usage
        for _col in _chunk.select_dtypes("float64").columns:
            _chunk[_col] = _chunk[_col].astype("float32")
        _chunks.append(_chunk)

    CRMS_continuous = pd.concat(_chunks)
    del _chunks
    gc.collect()

    print(CRMS_continuous.head(5))
    print(CRMS_continuous.shape)
    print(CRMS_continuous.dtypes)

    ### Step 1 ###########################################################
    print("Step 1: Make subsets")
    ######################################################################

    # Remove -H01 etc from Station ID
    CRMS_continuous["Station ID"] = CRMS_continuous["Station ID"].str.replace(
        "-H\d+", "", regex=True
    )
    # CRMS_continuous.head(10)

    output_name1 = "CRMS_Surface_salinity"
    output_name2 = "CRMS_Water_Elevation_to_Marsh"
    output_name3 = "CRMS_Water_Elevation_to_Datum"
    output_name4 = "CRMS_Water_Temp"

    # Create, save, and free each pivot table individually to reduce peak memory
    pivoted_salinity = CRMS_continuous.pivot_table(
        index=CRMS_continuous.index,
        columns="Station ID",
        values="Adjusted Salinity (ppt)",
    )
    pivoted_salinity.to_csv(os.path.join(Inputspace, output_name1 + file_suffix))
    pivoted_salinity.describe().to_csv(
        os.path.join(Processspace, "Surface_salinity_summary.csv")
    )
    del pivoted_salinity
    gc.collect()

    pivoted_W2m = CRMS_continuous.pivot_table(
        index=CRMS_continuous.index,
        columns="Station ID",
        values="Adjusted Water Elevation to Marsh (ft)",
    )
    pivoted_W2m.to_csv(os.path.join(Inputspace, output_name2 + file_suffix))
    pivoted_W2m.describe().to_csv(os.path.join(Processspace, "pivoted_W2m_summary.csv"))
    del pivoted_W2m
    gc.collect()

    pivoted_W2d = CRMS_continuous.pivot_table(
        index=CRMS_continuous.index,
        columns="Station ID",
        values="Adjusted Water Elevation to Datum (ft)",
    )
    pivoted_W2d.to_csv(os.path.join(Inputspace, output_name3 + file_suffix))
    pivoted_W2d.describe().to_csv(os.path.join(Processspace, "pivoted_W2d_summary.csv"))
    # Keep pivoted_W2d for the GEOID adjustment below

    pivoted_temp = CRMS_continuous.pivot_table(
        index=CRMS_continuous.index,
        columns="Station ID",
        values="Adjusted Water Temperature (°C)",
    )
    pivoted_temp.to_csv(os.path.join(Inputspace, output_name4 + file_suffix))
    pivoted_temp.describe().to_csv(
        os.path.join(Processspace, "pivoted_temp_summary.csv")
    )
    del pivoted_temp, CRMS_continuous
    gc.collect()

    ####### Additional modification #######
    print("Modify Geoid99 to Geoid12a/b for CRMS_Water_Elevation_to_Datum.csv")

    # Extract rows with Station IDs including "-H" which is Hydrographic hourly
    offsets_hydrographic = offsets.loc[
                           :, offsets.columns.str.contains("Date|-H01")
                           ].copy()
    offsets_hydrographic.shape
    offsets_hydrographic.index = pd.to_datetime(offsets.Date)
    offsets_hydrographic.drop(["Date"], axis=1, inplace=True)

    # Remove "-H" from column names
    offsets_hydrographic.columns = offsets_hydrographic.columns.str.replace(
        "-H\d+", "", regex=True
    )
    print(offsets_hydrographic.head())

    # Check if dataframe columns are duplicated or not

    if offsets_hydrographic.columns.duplicated().any():
        dup_cols = offsets_hydrographic.columns[
            offsets_hydrographic.columns.duplicated()
        ]
        for col in dup_cols:
            print(f"There are {col} duplicate columns in the dataframe:")
    else:
        print("There are no duplicate columns in the dataframe.")

    # Concentrate (merge) dataframes
    merged_df = pd.concat([offsets_hydrographic, pivoted_W2d], axis=0)
    merged_df.head(5)
    merged_df.to_csv(os.path.join(Processspace, "check_merged_df.csv"))

    # Drop columns where row 1 (offset value) is NaN
    offset_CRMSw2d = merged_df.loc[:, ~merged_df.iloc[0].isna()]
    offset_CRMSw2d.head(5)

    # Get the row number of Geoid12A/B start day
    geoid_row = offset_CRMSw2d.index.get_loc(pd.Timestamp(year=2013, month=10, day=1))

    # Modify the Geoid difference
    # get the row number of Geoid12A/B start day
    geoid_row = offset_CRMSw2d.index.get_loc(pd.Timestamp(year=2013, month=10, day=1))
    print("Before adjustment", offset_CRMSw2d.iloc[geoid_row - 3: geoid_row + 3, :])

    for i, col in enumerate(offset_CRMSw2d.columns):
        #    print (col)
        offset_CRMSw2d.iloc[:geoid_row, i] = round(
            offset_CRMSw2d.iloc[:geoid_row, i].add(offset_CRMSw2d.iloc[0, i]), 3
        )
    # offset_CRMSw2d.iloc[:,:]=round(offset_CRMSw2d.iloc[:,:]/3.28084,2) # convert into [m]

    print("After adjustment", offset_CRMSw2d.iloc[geoid_row - 3: geoid_row + 3, :])

    offset_CRMSw2d = offset_CRMSw2d.iloc[1:, :]  # delete first dummy row
    # offset_CRMSw2d.head(5)

    # Save a dataset (CSV)
    save_name = os.path.join(Inputspace, "CRMS_Geoid99_to_Geoid12b_offsets.csv")
    offset_CRMSw2d.to_csv(save_name)
    del offsets_hydrographic, merged_df, pivoted_W2d, offset_CRMSw2d, offsets
    gc.collect()

    # Calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print("Done Step 1")
    print("Time to Compute: \t\t\t", elapsed_time, " seconds")
    print("Job Finished ʕ •ᴥ•ʔ")

    pass
