#!/usr/bin/env python
# coding: utf-8
# CRMS2Resample
# Developed by the Center for Computation & Technology and Center for Coastal Ecosystem Design Studio at Louisiana State University (LSU).
# Developer: Jin Ikeda, Shu Gao, and Christopher E. Kees
# Last modified Aug 8, 2024

from CRMS_general_functions import *

def resample():
    ### Step 2 #############################################################################################################
    print ('Step 2: Resample and data-processing')
    ########################################################################################################################
    ### 2.1 Import modules ###
    # Import the general interpolation functions from the CRMS_general_functions.py file (some interpolation functions are keep inside of this code)

    start_time = time.time()

    # The target working directory
    # Workspace = "/Users/xxx/yyy/CRMS2Map/" # Local
    Workspace = os.getcwd() # Get the current working directory (same with the setup.py)

    # Print the current working directory
    print("Current working directory: {0}".format(os.getcwd()))

    Inputspace = os.path.join(Workspace, 'Input') # Make Input folder

    try:
        os.makedirs(Inputspace, exist_ok=True)

    except Exception as e:
        print(f"An error occurred while creating directories: {e}")


    ### 2.2 Open reference file
    # Input Subset type
    ########################################################################################################################
    # Data1: Salinity [ppt] or the data no editing is required.
    # Data2: Water_Elevation_to_Datum [ft] (CRMS_Geoid99_to_Geoid12a_offsets)
    # Data3: Water_Elevation_to_Marsh [ft]
    # Water_Elevation_to_Marsh data creates water depth and wet/dry boolean (dry:0 or wet:1)
    # Data4: Water Temperature (°C)
    ########################################################################################################################

    ########################################################################################################################
    # Get the CRMS files using the get_CRMS_file function
    Data_list = [1,2,3,4] # 1: Salinity, 2: Water_Elevation_to_Datum, 3: Water_Elevation_to_Marsh, 4: Water Temperature

    SLR_flag = False
    if SLR_flag:
        SLR = 30 # [cm] e.g. SLR in 2050 Sweet 2022

    for Data in Data_list:
        assert 1 <= Data <= 4,"Please input the values between 1-4 ...{{ (>_<) }}\n"

        file = get_CRMS_file(Data)
        file = os.path.join(Inputspace, file)
        file_name = file.split('.')[0]

        CRMS_subset = create_dataframe(file)

        start_year = CRMS_subset.index[0].year
        end_year = CRMS_subset.index[-1].year
        print(f'The {file_name} is from', start_year, 'to', end_year)

        ####################################################################################################################
        # User settings
        ####################################################################################################################
        # Modify the unit
        if (Data == 2 or Data == 3):
            CRMS_subset=round(CRMS_subset/3.28084,2) # convert into [m]
            if SLR_flag:
                CRMS_subset_SLR = CRMS_subset.add (SLR/10) # SLR in m
            print('unit is converted from [ft] to [m]')
        else:
            CRMS_subset[CRMS_subset > 100] = np.nan # Salinity sometimes exceeds 100 ppt due to drought but the data is unreliable and replaced by np.nan
            print('unit is unchanged, but an unreliable value over 100 ppt is replaced by np.nan')

        # Select the datasets between the two dates
        # CRMS_subset = CRMS_subset.query('index >= @start_date and index <= @end_date')

        ####################################################################################################################
        # 2.3 Resample and data-processing
        ####################################################################################################################
        file_name =file_name + '_' + str(start_year) + '_' + str(end_year)
        if (Data==3):
            file_name_list = [file_name]
            file_name2 = file_name + '_wdepth'
            file_name3 = file_name + '_wd'

            if SLR_flag:
                file_name4 = file_name + f'_SLR_{SLR}'
                file_name5 = file_name + f'_wdepth_SLR_{SLR}'
                file_name6 = file_name + f'_wd_SLR_{SLR}'
                file_name_list.extend([file_name2, file_name3, file_name4, file_name5, file_name6]) # SLR
            else:
                file_name_list.extend([file_name2, file_name3]) # no SLR

            # Edit inundation depth and percent time
            # CRMSw2d_depth = CRMS_subset.where(CRMS_subset <= 0, 0) # water depth
            CRMSw2d_depth = np.where(CRMS_subset <= 0, 0, np.where(CRMS_subset == np.nan ,np.nan,CRMS_subset)) # wet (1) or dry (o)
            CRMSw2d_depth = pd.DataFrame(CRMSw2d_depth, index=CRMS_subset.index, columns=CRMS_subset.columns)
            CRMSw2d_wd = np.where(CRMS_subset <= 0, 0, np.where(CRMS_subset > 0, 1, np.nan)) # wet (1) or dry (o)
            CRMSw2d_wd = pd.DataFrame(CRMSw2d_wd, index=CRMS_subset.index, columns=CRMS_subset.columns)
            if SLR_flag:
                CRMSw2d_depth_SLR = np.where(CRMS_subset_SLR <= 0, 0, np.where(CRMS_subset_SLR == np.nan ,np.nan,CRMS_subset_SLR))  # wet (1) or dry (o)
                CRMSw2d_depth_SLR = pd.DataFrame(CRMSw2d_depth_SLR, index=CRMS_subset_SLR.index, columns=CRMS_subset_SLR.columns)
                CRMSw2d_wd_SLR = np.where(CRMS_subset_SLR <= 0, 0, np.where(CRMS_subset_SLR > 0, 1, np.nan)) # wet (1) or dry (o)
                CRMSw2d_wd_SLR = pd.DataFrame(CRMSw2d_wd_SLR, index=CRMS_subset_SLR.index, columns=CRMS_subset_SLR.columns)
                datasets = [CRMS_subset, CRMSw2d_depth, CRMSw2d_wd, CRMS_subset_SLR, CRMSw2d_depth_SLR, CRMSw2d_wd_SLR]
            else:
                datasets = [CRMS_subset, CRMSw2d_depth, CRMSw2d_wd]
            print('water depth and hydro-period (dry:0 or wet:1) are added')

        elif (Data==2):
            file_name_list = [file_name]
            if SLR_flag:
                file_name2 = file_name + f'_SLR_{SLR}'
                file_name_list.extend([file_name2])
                file_name_list
                datasets = [CRMS_subset, CRMS_subset_SLR]

        else:
            file_name_list = [file_name]
            datasets = [CRMS_subset]

        for i in range(len(datasets)):
            ### datasets = [CRMS_subset, CRMSw2d_depth, CRMSw2d_wd] ###

            print(file_name_list[i])

            # Calculate hourly mean values
            hourly_mean=datasets[i].resample('h').mean()

            # Calculate daily mean values
            daily_mean=datasets[i].resample('D').mean()

            # Calculate Monthly mean values
            monthly_mean = daily_mean.resample('MS').mean().where(daily_mean.resample('MS').count() >= 5)                  # Use only include more than 5 days average data

            # Calculate Yearly mean values
            yearly_mean = monthly_mean.resample('YS').mean().where(monthly_mean.resample('YS').count() >= 9)               # Use only include more than 9 months average data

            # Provide the number of available stations and save file
            hourly_mean['num_station'] = hourly_mean.count(axis=1)
            save_name = os.path.join (Inputspace, file_name_list[i]+'_Hdata.csv')
            hourly_mean.to_csv(save_name)

            daily_mean['num_station'] = daily_mean.count(axis=1)
            save_name = os.path.join (Inputspace, file_name_list[i]+'_Ddata.csv')
            daily_mean.to_csv(save_name)

            monthly_mean['num_station']=monthly_mean.count(axis=1)
            save_name = os.path.join (Inputspace, file_name_list[i]+'_Mdata.csv')
            monthly_mean.to_csv(save_name)

            yearly_mean ['num_station']=yearly_mean.count(axis=1)                                                          # Provide the number of available stations
            save_name = os.path.join (Inputspace, file_name_list[i]+'_Ydata.csv')
            yearly_mean.to_csv(save_name)

    # Calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Done Step 2")
    print("Time to Compute: \t\t\t", elapsed_time, " seconds")
    print("Job Finished ʕ •ᴥ•ʔ")

if __name__ == "__main__":
    resample()
