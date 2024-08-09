#!/usr/bin/env python
# coding: utf-8
# CRMS_Discrete_Hydrographic for porewater salinity
# Developed by the Center for Computation & Technology and Center for Coastal Design Studio at Louisiana State University (LSU).
# Developer: Jin Ikeda, Shu Gao, and Christopher E. Kees
# Last modified Aug 9, 2024

from CRMS_general_functions import *


def subsets():
    ### Step 1 #############################################################################################################
    print("Step 1: Auto retrieve the original datasets")
    ########################################################################################################################
    ### 1.1 Import modules ###
    # Import the general interpolation functions from the CRMS_general_functions.py file (some interpolation functions are keep inside of this code)

    start_time = time.time()

    # The target working directory
    # Workspace = "/Users/xxx/yyy/CRMS2Map/" # Get the current working directory (same with the setup.py)
    Workspace = os.getcwd()  # HPC

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
    file_name = "CRMS_Discrete_Hydrographic"

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
    offsets_name = "GEOID99_TO_GEOID12A"

    file = file_name + file_suffix
    offsets_file = offsets_name + file_suffix

    try:
        CRMS = pd.read_csv(os.path.join(Inputspace, file), encoding="iso-8859-1")
        offsets = pd.read_csv(
            os.path.join(Inputspace, offsets_file), encoding="iso-8859-1"
        )
    except UnicodeDecodeError:
        # If the above fails due to an encoding error, try another encoding
        # print('encoding error')
        CRMS = pd.read_csv(os.path.join(Inputspace, file), encoding="utf-8")
        offsets = pd.read_csv(os.path.join(Inputspace, offsets_file), encoding="utf-8")

    print(CRMS.head(5))

    # Check data size
    print(CRMS.shape)

    # Check data type
    CRMS = pd.DataFrame(CRMS)
    print(CRMS.dtypes)

    # Combine the "Date (mm/dd/yyyy)" and "Time (hh:mm)". "Time Zone" is CST only (ignore)
    datetime_str = CRMS["Date (mm/dd/yyyy)"] + " " + CRMS["Time (hh:mm)"]

    # Convert the datetime string to a datetime object and set it as the index of the dataframe
    CRMS.index = pd.to_datetime(datetime_str, format="%m/%d/%Y %H:%M")
    CRMS.index.name = "Date"

    # Drop the columns that were used to create the index
    CRMS.drop(["Date (mm/dd/yyyy)", "Time (hh:mm)", "Time Zone"], axis=1, inplace=True)

    ### Step 1 ###########################################################
    print("Step 1: Make subsets")
    ######################################################################

    # Extract rows with CPRA Station IDs including "-P" which is porewater measurements
    CRMS_pore = CRMS.loc[CRMS["CPRA Station ID"].str.contains("-P")]

    # Check data size
    CRMS_pore.shape

    save_name = file_name + "_pore_origin.csv"
    CRMS_pore.to_csv(os.path.join(Inputspace, save_name))

    CRMS_pore.head(20)

    CRMS_pore = CRMS_pore.iloc[
        :, [0, 5, 14, 36, 37]
    ]  # Select CPRA Station ID, Measurement Depth (ft), Soil Porewater Salinity (ppt),Latitude, and Longitude

    # define the start and end dates
    # start_date = '2020-01-01'
    # end_date = '2022-01-01'

    # # select the datasets between the two dates
    # selected_CRMS_pore = CRMS_pore.query('index >= @start_date and index <= @end_date')

    # Remove -P01 -P02 etc from CPRA Station ID
    CRMS_pore["CPRA Station ID"] = CRMS_pore["CPRA Station ID"].str.replace(
        "-P\d+", "", regex=True
    )
    print(CRMS_pore.head(10))

    # Group by two different depth
    CRMS_pore["Measurement Depth (ft)"] = CRMS_pore["Measurement Depth (ft)"].round(3)
    CRMS_pore_group = CRMS_pore.groupby("Measurement Depth (ft)")
    depth_10_pore = CRMS_pore_group.get_group(0.328)  # 10 cm
    depth_30_pore = CRMS_pore_group.get_group(0.984)  # 30 cm

    # Create pivot table
    pivoted_10 = depth_10_pore.pivot_table(
        index=depth_10_pore.index,
        columns="CPRA Station ID",
        values="Soil Porewater Salinity (ppt)",
    )
    pivoted_30 = depth_30_pore.pivot_table(
        index=depth_30_pore.index,
        columns="CPRA Station ID",
        values="Soil Porewater Salinity (ppt)",
    )
    pivoted_10[pivoted_10 > 100] = np.nan
    pivoted_30[pivoted_30 > 100] = np.nan
    print("an unreliable value over 100 ppt is replaced by np.nan")

    output_name1 = "Pore_salinity_10"
    output_name2 = "Pore_salinity_30"
    save_name = output_name1 + file_suffix
    pivoted_10.to_csv(os.path.join(Inputspace, save_name))
    save_name = output_name2 + file_suffix
    pivoted_30.to_csv(os.path.join(Inputspace, save_name))

    pivoted_10.describe().to_csv(
        os.path.join(Processspace, "Pore_salinity_10_summary.csv")
    )
    pivoted_30.describe().to_csv(
        os.path.join(Processspace, "Pore_salinity_30_summary.csv")
    )

    ### Step 2 ###########################################################
    print("Step 2: Resample and data-processing")
    ######################################################################

    # resample the DataFrame to monthly frequency and calculate the mean value
    monthly_pore_10_mean = pivoted_10.resample("MS").mean()
    monthly_pore_10_mean["num_station"] = monthly_pore_10_mean.count(axis=1)
    save_name = output_name1 + "_Mdata.csv"
    monthly_pore_10_mean.to_csv(os.path.join(Inputspace, save_name))

    monthly_pore_30_mean = pivoted_30.resample("MS").mean()
    monthly_pore_30_mean["num_station"] = monthly_pore_30_mean.count(axis=1)
    save_name = output_name2 + "_Mdata.csv"
    monthly_pore_30_mean.to_csv(os.path.join(Inputspace, save_name))

    # resample the DataFrame to yearly frequency and calculate the mean value
    yearly_pore_10_mean = (
        monthly_pore_10_mean.resample("YS")
        .mean()
        .where(monthly_pore_10_mean.resample("YS").count() >= 3)
    )  # Use only include more than 9 months average data
    yearly_pore_10_mean["num_station"] = yearly_pore_10_mean.count(
        axis=1
    )  # Provide the number of available stations
    save_name = output_name1 + "_Ydata.csv"
    yearly_pore_10_mean.to_csv(os.path.join(Inputspace, save_name))
    # yearly_pore_10_mean.to_csv(save_name, na_rep=int(-99999))

    yearly_pore_30_mean = (
        monthly_pore_30_mean.resample("YS")
        .mean()
        .where(monthly_pore_30_mean.resample("YS").count() >= 3)
    )  # Use only include more than 9 months average data
    yearly_pore_30_mean["num_station"] = yearly_pore_30_mean.count(
        axis=1
    )  # Provide the number of available stations
    save_name = output_name2 + "_Ydata.csv"
    yearly_pore_30_mean.to_csv(os.path.join(Inputspace, save_name))
    # yearly_pore_30_mean.to_csv(save_name, na_rep=int(-99999))

    ### Step 3 ###########################################################
    print("Step 3: Interpolation")
    ######################################################################
    # code goes here

    ### Step 4 ###########################################################
    print("Step 4: 2D contour plots")
    ######################################################################
    # code goes here

    # Calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print("Done Step 1")
    print("Time to Compute: \t\t\t", elapsed_time, " seconds")
    print("Job Finished ʕ •ᴥ•ʔ")


if __name__ == "__main__":
    subsets()
