#!/usr/bin/env python
# coding: utf-8
# CRMS_Continuous_Hydrographic2subsets
# Developed by the Center for Computation & Technology and Center for Coastal Ecosystem Design Studio at Louisiana State University (LSU).
# Developer: Jin Ikeda, Shu Gao, and Christopher E. Kees
# Last modified Aug 8, 2024

from CRMS_general_functions import *
def subsets():

    ### Step 1 #############################################################################################################
    print("Step 1: Auto retrieve the original datasets")
    ########################################################################################################################
    ### 1.1 Import modules ###
    # Import the general interpolation functions from the CRMS_general_functions.py file (some interpolation functions are keep inside of this code)

    start_time = time.time()

    # The target working directory
    # Workspace = "/Users/xxx/yyy/CRMS2Map/" # Local
    Workspace = os.getcwd() # Get the current working directory (same with the setup.py)

    # Print the current working directory
    print("Current working directory: {0}".format(os.getcwd()))

    Inputspace = os.path.join(Workspace, 'Input') # Make Input folder
    Processspace = os.path.join(Workspace, 'Process') # Make Input folder

    try:
        os.makedirs(Inputspace, exist_ok=True)
        os.makedirs(Processspace, exist_ok=True)

    except Exception as e:
        print(f"An error occurred while creating directories: {e}")

    start_time = time.time()

    file_suffix=".csv"
    file_suffix_zip=".zip"
    file_name ="CRMS_Continuous_Hydrographic"

    ########################################################################################################################
    print("Download an original datasets")
    ########################################################################################################################
    # Define the name of the zip file and the csv file
    zip_file = file_name+file_suffix_zip
    csv_file = file_name+file_suffix
    print(csv_file)

    ### 1.2 Automatically download the file
    ########################################################################################################################
    # User can also manually download and upload the Continuous datasets from https://cims.coastal.la.gov/FullTableExports.aspx
    ########################################################################################################################

    url = "https://cims.coastal.la.gov/RequestedDownloads/ZippedFiles/" + file_name + file_suffix_zip
    download_CRMS(url, zip_file, csv_file,Inputspace) # Download the file

    print("Downloaded an original dataset")

    # Manually downloaded the offset conversion file
    offsets_name="GEOID99_TO_GEOID12A"

    file=file_name+file_suffix
    offsets_file =offsets_name+file_suffix

    try:
        CRMS = pd.read_csv(os.path.join(Inputspace,file), encoding='iso-8859-1')
        offsets = pd.read_csv(os.path.join(Inputspace,offsets_file), encoding='iso-8859-1')
    except UnicodeDecodeError:
        # If the above fails due to an encoding error, try another encoding
        # print('encoding error')
        CRMS = pd.read_csv(os.path.join(Inputspace,file), encoding='utf-8')
        offsets = pd.read_csv(os.path.join(Inputspace,offsets_file), encoding='utf-8')

    print(CRMS.head(5))

    # Check data size
    print(CRMS.shape)

    # Check data type
    CRMS=pd.DataFrame(CRMS)
    print(CRMS.dtypes)

    # Combine the "Date (mm/dd/yyyy)" and "Time (hh:mm)". "Time Zone" is CST only (ignore)
    datetime_str = CRMS["Date (mm/dd/yyyy)"] + ' ' + CRMS["Time (hh:mm:ss)"]

    # Convert the datetime string to a datetime object and set it as the index of the dataframe
    CRMS.index = pd.to_datetime(datetime_str, format='%m/%d/%Y %H:%M:%S')
    CRMS.index.name = "Date"

    # Drop the columns that were used to create the index
    CRMS.drop(["Date (mm/dd/yyyy)", "Time (hh:mm:ss)", "Time Zone"], axis=1, inplace=True)

    ### Step 1 ###########################################################
    print ('Step 1: Make subsets')
    ######################################################################

    # Extract rows with CPRA Station IDs including "-H" which is surface measurements -W: Well
    CRMS_continuous = CRMS.loc[CRMS['Station ID'].str.contains('-H')]

    # Check data size
    # print(CRMS_continuous.shape)

    # save_name=file_name+'_continuous_origin.csv'
    # CRMS_continuous.to_csv(save_name)
    # CRMS_continuous.head(20)

    CRMS_continuous = CRMS_continuous.iloc[:,[0,3,7,11,13,16,41,42]] #Station ID,Adjusted Temperature,Adjusted Salinity (ppt),Adjusted Water Elevation to Marsh (ft),Adjusted Water Elevation to Datum (ft)

    # Round down to nearest minute (#some data shows several seconds difference in the time index)
    CRMS_continuous.index = CRMS_continuous.index.floor('min')

    # Select the datasets between the two dates
    # start_date = '2020-01-01'
    # end_date = '2022-01-01'
    # selected_CRMS = CRMS.query('index >= @start_date and index <= @end_date')

    # Filter out rows where "Station ID" includes "-H0X" Hydrologic stations using surrogate data can be identified by the naming convention CRMSxxxx-H0X
    CRMS_continuous = CRMS_continuous[~CRMS_continuous['Station ID'].str.contains('-H0X')]

    # Remove -H01 etc from CPRA Station ID
    CRMS_continuous['Station ID'] = CRMS_continuous['Station ID'].str.replace('-H\d+', '', regex=True)
    # CRMS_continuous.head(10)

    # Create pivot table
    pivoted_salinity = CRMS_continuous.pivot_table(index=CRMS_continuous.index, columns='Station ID', values='Adjusted Salinity (ppt)')
    pivoted_W2m = CRMS_continuous.pivot_table(index=CRMS_continuous.index, columns='Station ID', values='Adjusted Water Elevation to Marsh (ft)')
    pivoted_W2d = CRMS_continuous.pivot_table(index=CRMS_continuous.index, columns='Station ID', values='Adjusted Water Elevation to Datum (ft)')
    pivoted_temp = CRMS_continuous.pivot_table(index=CRMS_continuous.index, columns='Station ID', values='Adjusted Water Temperature (°C)')

    output_name1='CRMS_Surface_salinity'
    output_name2='CRMS_Water_Elevation_to_Marsh'
    output_name3='CRMS_Water_Elevation_to_Datum'
    output_name4='CRMS_Water_Temp'

    save_name=output_name1+file_suffix
    pivoted_salinity.to_csv(os.path.join(Inputspace,save_name))

    save_name=output_name2+file_suffix
    pivoted_W2m.to_csv(os.path.join(Inputspace,save_name))

    save_name=output_name3+file_suffix
    pivoted_W2d.to_csv(os.path.join(Inputspace,save_name))

    save_name=output_name4+file_suffix
    pivoted_temp.to_csv(os.path.join(Inputspace,save_name))

    pivoted_salinity.describe().to_csv(os.path.join(Processspace,'Surface_salinity_summary.csv'))
    pivoted_W2m.describe().to_csv(os.path.join(Processspace,'pivoted_W2m_summary.csv'))
    pivoted_W2d.describe().to_csv(os.path.join(Processspace,'pivoted_W2d_summary.csv'))
    pivoted_temp.describe().to_csv(os.path.join(Processspace,'pivoted_temp_summary.csv'))

    ####### Additional modification #######
    print('Modify Geoid99 to Geoid12a for CRMS_Water_Elevation_to_Datum.csv')

    # Extract rows with Station IDs including "-H" which is Hydrographic hourly
    offsets_hydrographic = offsets.loc[:, offsets.columns.str.contains('Date|-H')].copy()
    offsets_hydrographic.shape
    offsets_hydrographic.index = pd.to_datetime(offsets.Date)
    offsets_hydrographic.drop(['Date'], axis=1, inplace=True)

    # Remove "-H" from column names
    offsets_hydrographic.columns = offsets_hydrographic.columns.str.replace('-H\d+', '', regex=True)
    print(offsets_hydrographic.head())

    # Check if dataframe columns are duplicated or not

    if offsets_hydrographic.columns.duplicated().any():
        dup_cols = offsets_hydrographic.columns[offsets_hydrographic.columns.duplicated()]
        for col in dup_cols:
            print(f"There are {col} duplicate columns in the dataframe:")
    else:
        print("There are no duplicate columns in the dataframe.")

    # Concentrate (merge) dataframes
    merged_df = pd.concat([offsets_hydrographic, pivoted_W2d], axis=0)
    merged_df.head(5)

    # Drop columns where row 1 (offset value) is NaN
    offset_CRMSw2d = merged_df.loc[:, ~merged_df.iloc[0].isna()]
    offset_CRMSw2d.head(5)

    # Get the row number of Geoid12A start day
    geoid_row = offset_CRMSw2d.index.get_loc(pd.Timestamp(year=2013, month=10, day=1))

    # Modify the Geoid difference
    # get the row number of Geoid12A start day
    geoid_row = offset_CRMSw2d.index.get_loc(pd.Timestamp(year=2013, month=10, day=1))
    print('Before adjustment', offset_CRMSw2d.iloc[geoid_row-3:geoid_row+3,:])

    for i, col in enumerate(offset_CRMSw2d.columns):
    #    print (col)
        offset_CRMSw2d.iloc[:geoid_row,i] = round(offset_CRMSw2d.iloc[:geoid_row,i].add(offset_CRMSw2d.iloc[0,i]),3)
    #offset_CRMSw2d.iloc[:,:]=round(offset_CRMSw2d.iloc[:,:]/3.28084,2) # convert into [m]

    print('After adjustment',offset_CRMSw2d.iloc[geoid_row-3:geoid_row+3,:])

    offset_CRMSw2d=offset_CRMSw2d.iloc[1:,:] # delete first dummy row
    #offset_CRMSw2d.head(5)

    #Save a dataset (CSV)
    save_name=os.path.join(Inputspace,'CRMS_Geoid99_to_Geoid12a_offsets.csv')
    offset_CRMSw2d.to_csv(save_name)

    # Calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print("Done Step 1")
    print("Time to Compute: \t\t\t", elapsed_time, " seconds")
    print("Job Finished ʕ •ᴥ•ʔ")

if __name__ == "__main__":
    subsets()

