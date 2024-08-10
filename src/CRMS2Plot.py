#!/usr/bin/env python
# coding: utf-8
# CRMS2Plot for time series analysis
# This file is developed for monthly data analysis. However, it includes hourly and daily analysis capabilities.
# Developed by the Center for Computation & Technology at Louisiana State University (LSU).
# Developer: Jin Ikeda
# Last modified Aug 8, 2024

### Functions ####################################################################################################
# Make a nested datasets for continuous data
def create_nested_datasets(file_name, file_name_o, file_suffix, threshold1,Discrete = False):

    datasets = {}  # monthly average dataset
    MA_datasets = {}  # moving average dictionaly
    for file_n, name_o in zip(file_name, file_name_o):
        file = file_n + file_suffix
        print(file)
        try:
            datasets[name_o] = pd.read_csv(file, encoding='utf-8')
        except UnicodeDecodeError:
            print('Encoding error')
            continue

        print(datasets[name_o].head(5))

        # Check data size and type
        print(datasets[name_o].shape, datasets[name_o].dtypes)
        datasets[name_o] = pd.DataFrame(datasets[name_o])

        # Delete columns where 'num_station' is lower than threshold1
        row_index = datasets[name_o].loc[datasets[name_o]['num_station'] < threshold1].index.tolist()
        datasets[name_o] = datasets[name_o].drop(row_index)

        # Move datetime into index
        datasets[name_o].index = pd.to_datetime(datasets[name_o].Date)

        # Drop the columns that were used to create the index
        datasets[name_o].drop(['Date'], axis=1, inplace=True)
        datasets[name_o].drop(['num_station'], axis=1, inplace=True)
        datasets[name_o] = datasets[name_o].iloc[2:,:]

        # Calculate moving average
        if Discrete == False:
            MA_datasets[name_o]= datasets[name_o].rolling(window=MA_window, center=True).mean() # , min_periods=9min_period is a rough criterion
        else:
            MA_datasets[name_o] = datasets[name_o].rolling(window=MA_window, center=True, min_periods=int(MA_window/2)).mean()  # , min_periods=9min_period is a rough criterion

        # Filtering the data
        datasets[name_o] = datasets[name_o].query('index >= @start_date and index <= @end_date')
        MA_datasets[name_o] = MA_datasets[name_o].query('index >= @start_date and index <= @end_date')

    return datasets, MA_datasets


def create_dataframe(file_name, date_column='Date'):

    # Use the appropriate pandas function to read the file
    if '.xlsx' in file_name:
        try:
            df = pd.read_excel(file_name)
        except UnicodeDecodeError:
            print('encoding error')            # CRMS = pd.read_csv(file, encoding='utf-8')
    elif '.csv' in file_name:
        try:
            df = pd.read_csv(file_name)
        except UnicodeDecodeError:
            print('encoding error')            # CRMS = pd.read_csv(file, encoding='utf-8')
    else:
        raise ValueError("Unsupported file type. The file must be a .xlsx or .csv file.")

    # Convert the DataFrame to the correct format
    df = pd.DataFrame(df)

    print(df.head(5))
    print(df.shape, df.dtypes)  # Check data size and type

    # Set the index to the 'Date' column and drop the original 'Date' column
    df.index = pd.to_datetime(df[date_column])
    df.drop([date_column], axis=1, inplace=True)

    # Future revision may consider filtering of datasets

    return df

def sub_dataframe_gen(sub_name,file_name):
    sub_datasets = {} # monthly average dataset

    for file, name_o in zip(sub_name,file_name):
        print(file)
        sub_datasets[name_o] = create_dataframe(file, 'Date')

    return sub_datasets


def get_sorted_files(base_path, file_name_pattern, sort_pattern):
    file_sub_name = []
    for sub_file in glob.glob(os.path.join(base_path, file_name_pattern)):
        if "_mean" not in sub_file and "_median" not in sub_file:
            file_sub_name.append(sub_file)
    return sorted(file_sub_name, key=lambda x: sort_basin.index(os.path.splitext(os.path.basename(x))[0].split(sort_pattern)[1]))


# For all duration and intervals dataset
def process_files(file_names, sort_basin, label):
    datasets = {label: {}}  # Initialize the nested dictionary with the label key
    for file_name in file_names:
        efg = []  # dummy list for appending
        df = pd.read_csv(file_name)  # read the HP file
        df2 = pd.read_csv(file_name.replace('HP', 'depth'))  # read the depth file
        df3 = df.merge(df2, on=df2.Date, how='left', suffixes=('_HP', '_depth'))
        df3 = df3.rename(columns={"key_0": "Date"})
        df3.index = pd.to_datetime(df3.Date)
        df3.drop(['Date', 'Date_HP', 'Date_depth'], axis=1, inplace=True)
        df3 = df3.reindex(columns=sorted(df3.columns))
        for k in np.arange(start=0, stop=df3.shape[1] + 1, step=2):
            efg.append(df3.iloc[:, k:k + 2].mean(axis=0, skipna=True).values.tolist())
        abc = pd.DataFrame(efg, columns=["HP_total", "Depth_total"])

        interval_dataframes = []
        column_names = []
        for year in range(start_year, end_year + 1, interval):
            mask = (df3.index.year >= year) & (df3.index.year < year + interval)
            interval_dataframes.append(df3[mask])
            column_names.extend([f"HP_{year}", f"Depth_{year}"])

        interval_efg = []
        for i, interval_df in enumerate(interval_dataframes):
            for k in np.arange(start=0, stop=interval_df.shape[1] + 1, step=2):
                interval_efg.extend(interval_df.iloc[:, k:k + 1].mean(axis=0, skipna=True).values.tolist())
            for k in np.arange(start=1, stop=interval_df.shape[1] + 1, step=2):
                interval_efg.extend(interval_df.iloc[:, k:k + 1].mean(axis=0, skipna=True).values.tolist())

        interval_efg = np.asarray(interval_efg)
        interval_efg = interval_efg.reshape(-1, (i+1)*2, order='F')
        interval_abc = pd.DataFrame(interval_efg, columns=column_names)
        all_abc = pd.concat([abc, interval_abc], axis=1)
        all_abc.dropna(axis=0, how='any', inplace=True)
        datasets[label][sort_basin[file_names.index(file_name)]] = all_abc

    return datasets


def plot_CRMS(datasets,MA_datasets,file_name_o,plot_range, plot_space): # ["Temp","Salinity","WL","W_depth","W_HP","WL_SLR","W_depth_SLR","W_HP_SLR"]
    plt.clf()
    fig, ax = plt.subplots(figsize=(6,3))
    ax.set_xlabel('Year')
    if file_name_o == "WL":
        ax.set_ylabel(u'Water level [NAVD88,m]')
        output = os.path.join(Photospace,'Water_level.png')
    elif file_name_o == "W_HP":
        ax.set_ylabel(u'Hydroperiod')
        output = os.path.join(Photospace,'Hydroperiod.png')
    elif file_name_o == "W_depth":
        ax.set_ylabel(u'Inundation depth [m]')
        output = os.path.join(Photospace,'Water_level.png')

    ax.plot(MA_datasets[file_name_o].index, MA_datasets[file_name_o].median(axis=1,skipna=True),'k--',linewidth=1)
    plt.fill_between(MA_datasets[file_name_o].index, MA_datasets[file_name_o].quantile(q=0.25,axis=1), MA_datasets[file_name_o].quantile(q=0.75,axis=1), alpha=0.9, linewidth=0, color='grey')
    plt.xlim(mp_dates.date2num(plot_period))
    plt.ylim(plot_range)
    major_ticks = np.arange(plot_range[0], plot_range[1]+0.01, plot_space)
    ax.set_yticks(major_ticks)
    plt.grid(color = 'k', linestyle = '--', linewidth = 0.1)
    plt.savefig(output,dpi=600,bbox_inches='tight')
    plt.show()
    plt.close()


def calculate_trends(df_MA, trend_columns,subdomain=False, sort_domain_list=None):
    trends = []
    boolean = []

    if subdomain:
        for i, j in enumerate(sort_domain_list):
            print(j)
            judge = adf_test(df_MA[trend_columns][j])
            boolean.append(judge)
            slopes, intercepts = linear_slope(df_MA[trend_columns][j])
            trends.append([slopes, intercepts])
    else:
        for i in df_MA.columns[trend_columns]:
            print(i)
            judge = adf_test(df_MA[i])
            boolean.append(judge)
            slopes, intercepts = linear_slope(df_MA[i])
            trends.append([slopes, intercepts])

    for i, boolean_value in enumerate(boolean):
        if boolean_value == 'False':
            trends[i] = [np.nan] * len(trends[i])

    return trends

def explore_map(polygon, point,Community):
    base = polygon.explore(
        column="BASIN",
        scheme="naturalbreaks",
        legend=True,
        k=10,
        tooltip=False,
        legend_kwds=dict(colorbar=False),
        name="Basin",
    )

    point.explore(
        m=base,
        color="red" if Community == "station" else None,
        column=None if Community == "station" else Community,
        scheme=None if Community == "station" else "naturalbreaks",
        legend=True,
        marker_kwds=dict(radius=5, fill=True),
        name=Community,
    )

    folium.LayerControl().add_to(base)

    return base

def create_subdomain_index(grouped_data, list_values):
    subdomain_index = {}
    key_list = []

    for name, data in grouped_data.items():
        subdomain_index[name] = {}

        for value in list_values:
            if value in data.groups:
                subdomain_index[name][value] = data.get_group(value)["CRMS_Sta"].values
                key_list.append(value)
            else:
                subdomain_index[name][value] = []

    return subdomain_index, key_list


def create_subdomain_datasets(datasets, file_name_o, subdomain_index, sort_list, output_path, subset_name):
    MA_subsets = {}  # moving average dictionary
    for i in file_name_o:
        print("Variable is \t", i)
        df_median = pd.DataFrame()  # empty dataframe
        MA_subsets[i] = {}  # moving average dictionary (nested dictionary)
        df_median_MA = pd.DataFrame()  # empty dataframe
        if i in ["W_HP", "W_depth", "W_HP_SLR", "W_depth_SLR"]:
            k = "W2M"  # this is the original data
        elif i == "WL_SLR":
            k = "WL"  # this is the original data
        else:
            k = i

        for j in sort_list:
            if i == "W_depth" and k =="W2M":
                print (i,j)
            df1 = datasets[i].loc[:, subdomain_index[k][j]]
            output_name1 = i + '_subset_' + subset_name + '_' + j + '.csv'
            df1.to_csv(os.path.join(output_path, output_name1))  # save each subset dataset for each variable

            # Calculate subset median
            median_values = df1.median(axis=1, skipna=True)
            num_available_columns = df1.count(axis=1)
            print(j, df1.shape, num_available_columns)
            median_values[num_available_columns < df1.shape[1] / 3] = np.nan  # Still think about the criterion.
            df_median[j] = median_values

            # Calculate moving average
            MA_subsets[i][j] = df1.rolling(window=MA_window, center=True).mean()  # we could add min_periods=9 min_period is a rough criterion
            # Calculate subset median for moving average
            median_values2 = MA_subsets[i][j].median(axis=1, skipna=True)
            num_available_columns2 = MA_subsets[i][j].count(axis=1)
            median_values2[num_available_columns2 < MA_subsets[i][j].shape[1] / 3] = np.nan  # Still think about the criterion.
            df_median_MA[j] = median_values2

            output_name2 = i + '_subset_' + subset_name + '_median.csv'
            df_median.to_csv(os.path.join(output_path, output_name2))
            output_name3 = i + '_subset_' + subset_name + '_median_MA.csv'
            df_median_MA.to_csv(os.path.join(output_path, output_name3))

    return MA_subsets

# For heatmap
def create_subdomain_correlation(datasets, file_name_o, sort_list, SST_basin, Q_data, subdomain_prcp, output_path, subset_name):
    subdomain_datasets = {}  # monthly average dataset
    corr_subdomain = {}

    for i in sort_list:
        print(i)
        df = pd.DataFrame()
        subdomain_datasets[i] = {}
        for j in file_name_o:
            subdomain_datasets[i][j] = datasets[j][i]
            df[j] = subdomain_datasets[i][j]

        df['GoM SST'] = SST_basin['GoM SST']
        if subset_name == 'basin' and (i == 'ME' or i == 'CS'):
            df['Q'] = Q_data.query('index >= @start_date and index <= @end_date')['cms']
        else:
            df['Q'] = SST_basin['AR_Q']
        df['Prcp'] = subdomain_prcp.query('index >= @start_date and index <= @end_date')[i].values
        df['U10'] = SST_basin['U10']
        df['V10'] = SST_basin['V10']
        df = df.rename(columns={'Temp': 'CRMS ST'})  # Change column name
        # column_reorder = list(range(5, df.shape[1])) + [1, 0, 2, 4, 3]  # column name reorder index
        # df_reorder = df.iloc[:, column_reorder]
        reorder_list =['GoM SST', 'Q', 'Prcp','U10', 'V10','CRMS ST','WL','Salinity', 'W_depth', 'W_HP']
        df_reorder = df.reindex(reorder_list, axis=1)
        output_name = subset_name + '_' + i + '.csv'
        df_reorder.to_csv(os.path.join(output_path, output_name))

        # only for heatmap
        df_reorder2 = df_reorder.copy()
        df_reorder2 = df_reorder2.rename(columns={'Q':'$\it{Q}$','Prcp': '$\it{P}$','U10':'$\it{U}$10','V10':'$\it{V}$10','Salinity': '$\it{S}$', 'WL': r'$\xi$'})
        df_reorder2.drop(['W_HP','W_depth'], axis=1, inplace=True)
        corr_subdomain[i] = df_reorder2.corr()

        output = os.path.join(Photospace,'basin_heat_' + i + '.png')
        heatplot(corr_subdomain[i], output)

    return corr_subdomain

def heatplot(correlation, output):
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(6, 6))

    # applying mask
    mask = np.triu(np.ones_like(correlation))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(correlation, mask=mask,annot=True,fmt=".2f",cmap="RdBu_r",vmin=-0.6, vmax=1) #mask=False

    # Display the heatmap plot
    plt.savefig(output,dpi=600,bbox_inches='tight')
    #plt.show()
    plt.close()

    #return plt.show()


def adf_test(MA_df):  # input muoving average
    # # ADF Test check the data is stationaly or not

    adft = adfuller(MA_df.dropna(), autolag="AIC")
    output_df = pd.DataFrame(
        {"Values": [adft[0], adft[1], adft[2], adft[3], adft[4]['1%'], adft[4]['5%'], adft[4]['10%']],
         "Metric": ["Test Statistics", "p-value", "No. of lags used", "Number of observations used",
                    "critical value (1%)", "critical value (5%)", "critical value (10%)"]})
    print(output_df)
    if adft[0] < adft[4]["5%"]:
        print("Reject Ho - Time Series is Stationary")
        judge = 'False'  # Stationary
    else:
        print("Failed to Reject Ho - Time Series is Non-Stationary")
        judge = 'True'  # Non-Stationary
    return judge


def linear_slope(df_MA):
    slope_MA = pd.DataFrame(df_MA.copy())
    # time_values = (df_MA.index - df_MA.index[0]).days/ 365.25
    slope_MA.index = (df_MA.index - pd.Timestamp("1981-1-1")) // pd.Timedelta(days=1) / 365.25
    # slope_MA.index=time_values
    slope_MA = slope_MA.dropna()
    # print(slope_MA)
    # dataset= pd.DataFrame(slope_MA[variable])
    # dataset=dataset.dropna()

    # Fit a linear regression model to the data
    slope, intercept, r_value, p_value, std_err = stats.linregress(slope_MA.index, slope_MA.iloc[:, 0])

    # The 'slope' variable now contains the slope of the linear regression line
    print("Slope:", slope, "Intercept", intercept, "R_value", r_value, "p_value", p_value)

    return slope, intercept

########################################################################
print ('Data_anaysis')
######################################################################

### 1.1 Import modules ###
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import glob
from datetime import datetime, timedelta
# stats
from statsmodels.tsa.stattools import adfuller  
from scipy import stats
# plot
from matplotlib import pyplot as plt
import matplotlib.dates as mp_dates
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import itertools

# The target working directory

# Make Output folder
Workspace = "C:/Users/jikeda/Desktop/CRMS2Map/CRMS_devtest"

# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))

# Change the current working directory
os.chdir(Workspace)

Photospace = os.path.join(Workspace, 'Photo')
#Output_space=os.path.join(Workspace,'bootstrap_Output')

try:
    os.mkdir(Photospace)
except:
    pass

### Parameters ###########################################################
start_date = '2008-01-01' # 'yyyy-mm-dd'
start_date_climate = '1981-01-01' # 'yyyy-mm-dd'
end_date = '2024-02-29'   # 'yyyy-mm-dd'
threshold1 = 200 # Delete columns where 'num_station' is lower than the threshold for continuous data
threshold2 = int(threshold1/2) # Delete columns where 'num_station' is lower than the threshold for discrete data

end_date_datetime = datetime.strptime(end_date, '%Y-%m-%d') # Convert the end_date string to a datetime object
end_date_datetime = end_date_datetime + timedelta(days=1) # Add one day to the end_date
end_date_plus_one = end_date_datetime.strftime('%Y-%m-%d') # Convert the datetime object back to a string

# Update the plot_period
plot_period = [start_date, end_date_plus_one] # 'yyyy-mm-dd'
print('plot_period',plot_period)

Sub_basinspace = os.path.join(Workspace, 'Sub_basin') # Make Sub_basin folder
Sub_marshspace = os.path.join(Workspace, 'Sub_marsh') # Make Sub_marsh folder

try:
    #os.makedirs(Photpspace, exist_ok=True)
    os.makedirs(Sub_basinspace, exist_ok=True)
    os.makedirs(Sub_marshspace, exist_ok=True)

except Exception as e:
    print(f"An error occurred while creating directories: {e}")

#############################################
# color_palette for plots
#############################################
# Create ramdom colors
#p.random.seed(42)  # Set the seed for reproducibility
color_palette = []

color_palette.append([153/255, 0, 0]) # Pontchartrain
color_palette.append([255/255, 200/255, 0/255]) # Breton Sound
color_palette.append([255/255, 153/255, 0/255]) # Mississippi River Delta
color_palette.append([204/255, 210/255, 0/255]) # Barataria
color_palette.append([0/255, 128/255, 0/255]) # Terrebonne
color_palette.append([0/255, 0/255, 255/255]) # Atchafalaya
color_palette.append([153/255, 153/255, 255/255]) # Teche/Vermilion
color_palette.append([204/255, 102/255, 153/255]) # Mermentau
color_palette.append([255/255, 0, 255/255]) # Calcasieu/Sabine

color_palette_vegetation = []
#print(color_palette)
color_palette_vegetation.append([230/255, 230/255, 0]) # Brackish
color_palette_vegetation.append([115/255, 255/255, 223/255]) # Freshwater
color_palette_vegetation.append([223/255, 115/255, 255/255]) # Intermediate
color_palette_vegetation.append([255/255, 85/255, 0/255]) # Saline
color_palette_vegetation.append([204/255, 204/255, 204/255]) # Swamp

### Step 2 ###########################################################
print ('Step 2: Read input data ')
######################################################################

### 2.1 Read CRMS files ###

file_suffix=".csv"

### Open continuous files
file_name1="CRMS_Water_Temp_2006_2024_Mdata"
file_name2="CRMS_Surface_salinity_2006_2024_Mdata"
file_name3="CRMS_Geoid99_to_Geoid12a_offsets_2006_2024_Mdata"
file_name4="CRMS_Water_Elevation_to_Marsh_2006_2024_wdepth_Mdata"
file_name5="CRMS_Water_Elevation_to_Marsh_2006_2024_wd_Mdata"


file_name=[file_name1,file_name2,file_name3,file_name4,file_name5]
file_name_o=["Temp","Salinity","WL","W_depth","W_HP"]

### Open discrete files
file_name9="Pore_salinity_10_Mdata"
file_name10="Pore_salinity_30_Mdata"
file_name_discrete=[file_name9,file_name10]
file_name_o_discrete=["Pore_10","Pore_30"]

#######################################################################################################################
# Set a moving window range for yearly analysis
# H: hourly, D: daily, M: monthly

if "Mdata" in file_name1:
    MA_window = 12
elif "Ddata" in file_name1:
    MA_window = 365
elif "Hdata" in file_name1:
    MA_window = 8766
else:
    raise ValueError("The file name must contain one of 'Mdata', 'Ddata' or 'Hdata'")
#######################################################################################################################

### 2.2 Open climate file

file_suffix2=".xlsx"
file_name11="MonthlySST"
file_name11=file_name11+file_suffix2

# Create dataframe
SST = create_dataframe(file_name11,'Date')
SST.drop(['GI_trend'], axis=1, inplace=True)
# Calculate moving average
MA_datasets_SST= SST.rolling(window=MA_window, center=True).mean() # , min_periods=9min_period is a rough criterion

# Calcasieu River data
path_folder5 = "C:/Users/jikeda/Desktop/Time_Series_Analysis_version2/River_discharge/Time_series_analysis/"


CS_Q_file=os.path.join(path_folder5,'CS_discharge_since_2008.csv') # Calcasieu River Near Oberlin
CS_Q = create_dataframe(CS_Q_file,'Date')

# Create a nested dataset for continuous data between 2008-2022
datasets, MA_datasets = create_nested_datasets(file_name, file_name_o, file_suffix, threshold1)

MA_datasets["WL"].to_excel('MA_timeseris_WL.xlsx') # Save the moving average dataset for analyzing a spatial mapping paper

print('##########################################################################################################################\n')
print('W_HP datasets', datasets["W_HP"])
print('\n##########################################################################################################################\n')

# Display stats
print("HP =",datasets["W_HP"].mean().mean(),", Depth= ",datasets["W_depth"].mean().mean())

# Create a nested dataset for discrete data between 2008-2022
datasets_discrete, MA_datasets_discrete = create_nested_datasets(file_name_discrete, file_name_o_discrete, file_suffix, threshold2, Discrete=True)

MA_datasets_discrete["Pore_10"].to_csv('Pore_10_MA.csv')
datasets_discrete["Pore_10"].to_csv('Pore_10.csv')

print('##########################################################################################################################\n')
print('W_HP datasets', datasets_discrete["Pore_10"].head(10))
print('\n##########################################################################################################################\n')


### Step 3 ###########################################################
print('##########################################################################################################################\n')
print ('Step 3: Plot input data ')
print('\n##########################################################################################################################\n')
######################################################################

# # Plot CRMS data
# # Future modification: Jin 01/27/24
# #
# # Check water level

fig, ax = plt.subplots(figsize=(6,3))
# ax.plot(MA_datasets["WL"].index, MA_datasets["WL"].median(axis=1),'o', mfc='none', mec='k')

ax.set_xlabel('Year')
ax.set_ylabel(r'$\xi$ [m,NAVD88]')
ax.plot(MA_datasets["WL"].index, MA_datasets["WL"].median(axis=1),'k--',linewidth=1)
plt.fill_between(MA_datasets["WL"].index, MA_datasets["WL"].quantile(q=0.25,axis=1), MA_datasets["WL"].quantile(q=0.75,axis=1), alpha=0.9, linewidth=0, color='grey')

#plt.xlim([2007, 2023])
plt.xlim(mp_dates.date2num(plot_period))
plt.ylim([0, 0.4])
major_ticks = np.arange(0, 0.41, 0.1)
#minor_ticks = np.arange(0, 0.31, 1)
ax.set_yticks(major_ticks)
#ax.set_yticks(minor_ticks)
plt.grid(color = 'k', linestyle = '--', linewidth = 0.1)
output = os.path.join(Photospace, 'Water_level.png')
plt.savefig(output,dpi=600,bbox_inches='tight')
#plt.show()
plt.close()

# # Check hydro-period
# plt.clf()
# fig, ax = plt.subplots(figsize=(6,3))
# ax.plot(MA_datasets["W_HP"].index, MA_datasets["W_HP"].median(axis=1),'o', mfc='none', mec='k')
#
# ax.set_xlabel('Year')
# ax.set_ylabel('Hydroperiod')
# ax.plot(MA_datasets["W_HP"].index, MA_datasets["W_HP"].median(axis=1),'k--',linewidth=1)
# plt.fill_between(MA_datasets["W_HP"].index, MA_datasets["W_HP"].quantile(q=0.25,axis=1), MA_datasets["W_HP"].quantile(q=0.75,axis=1), alpha=0.9, linewidth=0, color='grey')
#
# #plt.xlim([2007, 2023])
# plt.ylim([0, 1])
# plt.grid(color = 'k', linestyle = '--', linewidth = 0.1)
# output = os.path.join(Photospace, 'Hydro_period.png')
# plt.savefig(output,dpi=600,bbox_inches='tight')
# plt.show()
# plt.close()
#
# # Check pore-water
# plt.clf()
# # create plot
# fig, ax = plt.subplots(figsize=(6,3))
# ax.set_xlabel('Year')
# ax.set_ylabel('Salinity [ppt]')
# ax.plot(MA_datasets["Salinity"].index, MA_datasets["Salinity"].median(axis=1,skipna=True),'k--',linewidth=1)
# ax.plot(MA_datasets_discrete["Pore_10"].index, MA_datasets_discrete["Pore_10"].median(axis=1,skipna=True),'g--',linewidth=1)
# ax.plot(MA_datasets_discrete["Pore_30"].index, MA_datasets_discrete["Pore_30"].median(axis=1,skipna=True),'r--',linewidth=1)
#
# plt.fill_between(MA_datasets_discrete["Pore_30"].index, MA_datasets_discrete["Pore_30"].quantile(q=0.25,axis=1), MA_datasets_discrete["Pore_30"].quantile(q=0.75,axis=1), alpha=0.5, linewidth=0, color='r')
# plt.fill_between(MA_datasets_discrete["Pore_10"].index, MA_datasets_discrete["Pore_10"].quantile(q=0.25,axis=1), MA_datasets_discrete["Pore_10"].quantile(q=0.75,axis=1), alpha=0.5, linewidth=0, color='g')
# plt.fill_between(MA_datasets["Salinity"].index, MA_datasets["Salinity"].quantile(q=0.25,axis=1), MA_datasets["Salinity"].quantile(q=0.75,axis=1), alpha=0.9, linewidth=0, color='grey')
#
#
# plt.xlim(mp_dates.date2num(plot_period))
# #ax.xaxis.set_major_formatter(mp_dates.DateFormatter('%Y'))
# plt.ylim([0, 16])
# major_ticks = np.arange(0, 17, 4)
# #minor_ticks = np.arange(0, 0.31, 1)
# ax.set_yticks(major_ticks)
# #ax.set_yticks(minor_ticks)
# plt.grid(color = 'k', linestyle = '--', linewidth = 0.1)
# ax.legend(['Surface', 'Pore d10', 'Pore d30'])
# output = os.path.join(Photospace, 'Salinity.png')
# plt.savefig(output,dpi=600,bbox_inches='tight')
# plt.show()
# plt.close()
#
# # Make CRMS plots
# # plot_CRMS(datasets, MA_datasets,"WL",[0, 0.4,], 0.1)
# # plot_CRMS(datasets, MA_datasets,"W_HP",[0, 1,], 0.2)
# # plot_CRMS(datasets, MA_datasets,"W_depth",[0, 0.3,], 0.1)
#
# # Temperature
#
# # plt.clf()
# # create plot
# fig, ax = plt.subplots(figsize=(6,3))
#
# ax.set_xlabel('Year')
# ax.set_ylabel(u'Temp [℃]')
# lab1,= ax.plot(MA_datasets["Temp"].index, MA_datasets["Temp"].median(axis=1,skipna=True),'k--',linewidth=1,label='CRMS')
# lab2,= ax.plot(MA_datasets_SST.index, MA_datasets_SST.SST,'g--',linewidth=1, label='GoM')
# plt.fill_between(MA_datasets["Temp"].index, MA_datasets["Temp"].quantile(q=0.25,axis=1), MA_datasets["Temp"].quantile(q=0.75,axis=1), alpha=0.9, linewidth=0, color='grey')
#
# #plt.xlim(mp_dates.date2num(['2000-01-01', '2023-01-01']))
# plt.xlim(mp_dates.date2num(plot_period))
# plt.ylim([20, 28])
# major_ticks = np.arange(20, 31, 2)
# major_ticks2 = np.arange(0, 16000,3000)
# #minor_ticks = np.arange(20, 31, 1)
# plt.grid(color = 'k', linestyle = '--', linewidth = 0.1)
#
#
# # # Adding Twin Axes to plot using dataset_2
# # ax2 = ax.twinx()
#
# # color = 'blue'
# # ax2.set_ylabel('Q [cms]')
# # lab3,=ax2.plot(MA_datasets_SST.index, MA_datasets_SST.AR_Q,color = color,label='AR Q')
# # ax2.tick_params(axis ='y')
# # ax2.set_ylim([0, 15000])
# # ax.set_yticks(major_ticks)
# # ax2.set_yticks(major_ticks2)
# #ax.set_yticks(minor_ticks)
# ax.legend(handles=[lab1, lab2],loc='upper left')
# output = os.path.join(Photospace, 'Temp.png')
# plt.savefig(output,dpi=600,bbox_inches='tight')
# plt.show()
# plt.close()
#
#
# # Precipitation and river discharge
#
# plt.clf()
# # create plot
# fig, ax = plt.subplots(figsize=(6,3))
#
# ax.set_xlabel('Year')
# ax.set_ylabel(u'Prcp [mm/km2]')
# lab3,=ax.plot(MA_datasets_SST.index, MA_datasets_SST.Prcp,color = 'c',label='Prcp')
#
# #plt.xlim(mp_dates.date2num(['2000-01-01', '2023-01-01']))
# plt.xlim(mp_dates.date2num(plot_period))
# plt.ylim([0, 500])
# major_ticks = np.arange(0, 501, 100)
# major_ticks2 = np.arange(0, 16000,3000)
# #minor_ticks = np.arange(20, 31, 1)
# plt.grid(color = 'k', linestyle = '--', linewidth = 0.1)
#
#
# # Adding Twin Axes to plot using dataset_2
# ax2 = ax.twinx()
#
# color = 'blue'
# ax2.set_ylabel('Q [cms]')
# lab4,=ax2.plot(MA_datasets_SST.index, MA_datasets_SST.AR_Q,color = color,label='AR Q')
# ax2.tick_params(axis ='y')
# ax2.set_ylim([0, 15000])
# ax.set_yticks(major_ticks)
# ax2.set_yticks(major_ticks2)
# #ax.set_yticks(minor_ticks)
# ax.legend(handles=[lab3,lab4],loc='upper left')
# output = os.path.join(Photospace, 'Fresh_water.png')
# plt.savefig(output,dpi=600,bbox_inches='tight')
# plt.show()
# plt.close()
#
#
# # Winds
#
# plt.clf()
# # create plot
# fig, ax = plt.subplots(figsize=(6,3))
#
# ax.set_xlabel('Year')
# ax.set_ylabel(u'u [m/s]')
# lab3,=ax.plot(MA_datasets_SST.index, MA_datasets_SST.U10,color = 'g',label='U10')
# lab4,=ax.plot(MA_datasets_SST.index, MA_datasets_SST.V10,color = 'orange',label='V10')
# lab5,=ax.plot(MA_datasets_SST.index, MA_datasets_SST.UV,color = 'k',label='Speed')
# #plt.xlim(mp_dates.date2num(['2000-01-01', '2023-01-01']))
# plt.xlim(mp_dates.date2num(plot_period))
# plt.ylim([-4, 4])
# major_ticks = np.arange(-4, 5, 2)
# #major_ticks2 = np.arange(0, 16000,3000)
# #minor_ticks = np.arange(20, 31, 1)
# plt.grid(color = 'k', linestyle = '--', linewidth = 0.1)
#
#
# # Adding Twin Axes to plot using dataset_2
# #ax2 = ax.twinx()
#
# #color = 'blue'
# #ax2.set_ylabel('Q [cms]')
# #lab4,=ax2.plot(MA_datasets_SST.index, MA_datasets_SST.AR_Q,color = color,label='AR Q')
# #ax2.tick_params(axis ='y')
# #ax2.set_ylim([0, 15000])
# ax.set_yticks(major_ticks)
# ax2.set_yticks(major_ticks2)
# #ax.set_yticks(minor_ticks)
# ax.legend(handles=[lab3,lab4, lab5],loc='lower left')
# output = os.path.join(Photospace, 'wind.png')
# plt.savefig(output,dpi=600,bbox_inches='tight')
# plt.show()
# plt.close()


### Step 4 ###########################################################
print('##########################################################################################################################\n')
print ('Step 4: check trends ')
print('\n##########################################################################################################################\n')
######################################################################
#
df_MA = MA_datasets_SST.copy() # combine MA climate data and CRMS data
columns = ["Temp", "Salinity", "WL", "W_HP", "W_depth"]

for col in columns:
    df_MA[col] = MA_datasets[col].median(axis=1, skipna=True)
    df_MA[col+"_Q1"] = MA_datasets[col].quantile(q=0.25, axis=1)
    df_MA[col+"_Q3"] = MA_datasets[col].quantile(q=0.75, axis=1)

df_MA = df_MA[df_MA.index.notnull()]

long_trends=calculate_trends(df_MA, slice(6)) # 1981 - 2022 for 5 variables
print('##########################################################################################################################\n')
print("\n\n","Long-term trends for climate driver is",long_trends)
print('\n##########################################################################################################################\n')

# This is a plot data for paper
# Convert the datetime index to modified Julian date (number of days since November 17, 1858) For N_Graph but sligtly change the referenced days

df_MA_temp = df_MA.copy().query('index <= @end_date')
print (df_MA_temp)
df_MA_temp.to_excel('MA_timeseris.xlsx')
df_MA_temp.fillna('=', inplace=True)
#df_MA.index = (df_MA.index - pd.Timestamp("1858-11-17")) // pd.Timedelta(days=1)
df_MA_temp.index = (df_MA_temp.index - pd.Timestamp(start_date_climate)) // pd.Timedelta(days=1)
output_name = 'MA_timeseris_plot.xlsx'
df_MA_temp.to_excel(output_name)
print(df_MA_temp.tail())

df_MA=df_MA.query('index >= @start_date and index <= @end_date') # 2008 - 2022
df_MA_corr=df_MA.copy()
mid_trends=calculate_trends(df_MA, slice(None)) 
print('##########################################################################################################################\n')
print("\n\n","Mid-term trends (during observation ) is",mid_trends)
print('\n##########################################################################################################################\n')

### Step 5 ###########################################################
print('##########################################################################################################################\n')
print ('Step 5: check correlations ')
print('\n##########################################################################################################################\n')
######################################################################

corr=df_MA_corr.corr()
print ('The correlation of Moving average',corr)

# Merge two datasets
for i in columns:
    SST[i] = np.nan # Create a column
    print(i)
    start_row = SST.index.get_loc(datasets[i].index[0])
    end_row = SST.index.get_loc(datasets[i].index[-1])
    print (start_row, end_row)
    SST.iloc[start_row:end_row+1,SST.columns.get_loc(i)] = datasets[i].median(axis=1,skipna=True) # each datasets have different length

SST.head()
SST.to_excel('monthly_median_for_correlation.xlsx')
#
SST = SST.rename(columns={'SST': 'GoM SST',
                                      'Temp': 'CRMS ST','W_HP': 'HP','W_depth':'ID'})  # Change column name
SST_plot=SST.copy()
SST_plot.drop(['UV','ID'], axis=1, inplace=True)
SST_plot=SST_plot.query('index >= @start_date and index <= @end_date')
corr2=SST_plot.corr()
output = os.path.join(Photospace, 'correlation.png')
heatplot(corr2, output)

### Step 6 ###########################################################
print('##########################################################################################################################\n')
print ('Step 6: Analyze subdomain and vegetation')
print('\n##########################################################################################################################\n')
######################################################################

# The target grab point file
path_folder2 = "C:/Users/jikeda/Desktop/CRMS2Map/Code_dev/Input"
path_folder3 = "C:/Users/jikeda/Desktop/CRMS2Map/Code_dev/Output"
path_folder4 = "C:/Users/jikeda/Desktop/Time_Series_Analysis_version2/CRMS/CRMS_Marsh_Vegetation/"

polygon_file=os.path.join(path_folder2,'Basin_NAD83.shp') # 10 basin provided by CPRA
Basin_community_file=os.path.join(path_folder4,'CRMS_station_Basin_Community.shp') # 5 vegetation (marsh) community analyzed by Jin

Temp_point_file=os.path.join(path_folder3,'CRMS_stations_Water_Temp.shp')
Salinity_point_file=os.path.join(path_folder3,'CRMS_stations_Surface_salinity.shp')
W2l_point_file=os.path.join(path_folder3,'CRMS_stations_Water_Elevation_to_Datum.shp')
W2m_point_file=os.path.join(path_folder3,'CRMS_stations_Water_Elevation_to_Marsh.shp') # don't know why 'CRMS0287' is included in CRMS2plot code (12/19/23). Temporary ,manually delete 'CRMS0287'
#
### 6.1 Open polygon and basin community files
polygon=gpd.read_file(polygon_file)
Basin_community=gpd.read_file(Basin_community_file)

Temp_point = gpd.read_file(Temp_point_file)
Salinity_point=gpd.read_file(Salinity_point_file)
W2l_point = gpd.read_file(W2l_point_file)
W2m_point = gpd.read_file(W2m_point_file)
W2m_point = W2m_point[W2m_point.CRMS_Sta != 'CRMS0287']

subset_file=[Temp_point,Salinity_point,W2l_point,W2m_point]
subset_file_name=["Temp","Salinity","WL","W2M"]


Subbasinspace = os.path.join(Workspace, 'Sub_basin')
Submarshspace = os.path.join(Workspace, 'Sub_marsh')
Subbasinspace_SLR = os.path.join(Workspace, 'Sub_basin_SLR')

try:
    os.mkdir(Subbasinspace, exist_ok=True)
    os.mkdir(Submarshspace, exist_ok=True)
    os.mkdir(Subbasinspace_SLR, exist_ok=True)
except:
    pass

# Convert to km2
polygon['Square_km']=polygon['ACRES']*0.00404686
print(polygon)

# Check the long and lat
print(W2m_point)

### This is an optional on jupyterhub
import folium

# base = explore_map(polygon, W2m_point,"station")
# base
# base2 = explore_map(polygon, Basin_community,"Community")
# base2

print('##########################################################################################################################\n')
print('Grouped by sub_basin')
print('##########################################################################################################################\n')

subsets = {} # add basin information on sub dataset
grouped_data = {} # grouped data in each subset

for file, name in zip(subset_file, subset_file_name):
    print(name)
    try:
        subsets[name] = gpd.sjoin(file,polygon,how='inner',predicate='within')
        #output_subsets =name + "_subset.csv"
        #subsets[name].to_csv(output_subsets)
        grouped_data[name] = subsets[name].groupby('BASIN')

    except UnicodeDecodeError:
        # If the above fails due to an encoding error, try another encoding
        print('Encoding error')

basin_list=set(subsets["Temp"].BASIN.values)
print(basin_list) # check data
# print(grouped_data)

# Create a index list for sub_basin dataset
basin_index, basin_key = create_subdomain_index(grouped_data, basin_list)
print (basin_index.keys())
print (basin_index['W2M']['BA']) # check the stations

sort_basin=['PO','BS','MR','BA','TE','AT','TV','ME','CS'] # 'Perl basin is very small and exclude from analysis'

MA_subsets = create_subdomain_datasets(datasets, file_name_o, basin_index, sort_basin, Subbasinspace, 'basin')

#############################################
# Make median datasets for each domain and variable
#############################################
file1 = Subbasinspace +'/*_median.*csv'

file_sub_name1=[]
file_sub_name_SLR=[]

for sub_file in glob.glob(file1):
    if "_SLR_" not in sub_file:
        print(sub_file)
        file_sub_name1.append(sub_file)
    else:
        print(sub_file)
        file_sub_name_SLR.append(sub_file)

file2 = Subbasinspace +'/*_median_MA.*csv'
file_sub_name2=[]

for sub_file in glob.glob(file2):
    if "_SLR_" not in sub_file:
        print(sub_file)
        file_sub_name2.append(sub_file)

sorted_file_name_o=['Salinity', 'Temp', 'WL', 'W_depth', 'W_HP'] # need to reorder the variables
print(file_sub_name1)

subdatasets=sub_dataframe_gen(file_sub_name1,sorted_file_name_o)
subdatasets_SLR=sub_dataframe_gen(file_sub_name_SLR,file_name_o[-3:])
MA_subdatasets=sub_dataframe_gen(file_sub_name2,sorted_file_name_o)

#############################################
# Display correlation plots
#############################################

# corr={}
# for i in sorted_file_name_o:
#     print(i)
#     corr[i]=subdatasets[i].corr()
#
#     output = os.path.join(Photospace, 'sub_'+ i+ '.png')
#     heatplot(corr[i], output)

#############################################
# Display basin level plots
#############################################

# Temp

plt.clf()
fig, ax = plt.subplots(figsize=(6,3))
ax.set_xlabel('Year')
ax.set_ylabel('$\it{T}$ [℃]')
lab=[]
for i,j in enumerate(sort_basin):
    ax.plot(MA_subdatasets["Temp"].index, MA_subdatasets["Temp"][j],color=color_palette[i],linewidth=1,label=j)
plt.xlim(mp_dates.date2num(plot_period))
plt.ylim([15, 30])
major_ticks = np.arange(15, 31, 5)
ax.set_yticks(major_ticks)
plt.grid(color = 'k', linestyle = '--', linewidth = 0.1)
B,C =ax.get_legend_handles_labels()
ax.legend(C,loc='lower right') # Rabeling

B,C =ax.get_legend_handles_labels()
ax.legend(C,loc='lower right') # Rabeling
output = os.path.join(Photospace, 'Temp_basin.png')
plt.savefig(output,dpi=600,bbox_inches='tight')
#plt.show()
plt.close()

# Salinity

fig, ax = plt.subplots(figsize=(6,3))
ax.set_xlabel('Year')
ax.set_ylabel('$\it{S}$ [ppt]')
#
for i,j in enumerate(sort_basin):
    ax.plot(MA_subdatasets["Salinity"].index, MA_subdatasets["Salinity"][j],color=color_palette[i],linewidth=1,label=j)

plt.xlim(mp_dates.date2num(plot_period))
plt.ylim([0, 20])
major_ticks = np.arange(0, 21, 4)
#minor_ticks = np.arange(0, 0.31, 1)
ax.set_yticks(major_ticks)
plt.grid(color = 'k', linestyle = '--', linewidth = 0.1)
B,C =ax.get_legend_handles_labels()
ax.legend(C,loc='upper right',ncol=2) # Rabeling
output = os.path.join(Photospace, 'Salinity_basin.png')
plt.savefig(output,dpi=600,bbox_inches='tight')
#plt.show()
plt.close()

trends= calculate_trends(MA_subdatasets, "Salinity",subdomain=True, sort_domain_list=sort_basin)
print("\n\n","CRMS trends for climate driver is",trends)

##############################################################################################
# correlation with AR_Q

correlations=[]
for col in MA_subdatasets["Salinity"].columns:  # Transpose to iterate over columns
    correlation = df_MA_corr.AR_Q.corr(MA_subdatasets["Salinity"][col])
    correlations.append(correlation)
print('The correlation between AR_Q and Salinity',correlations) # sort_basin=['PO','BS','MR','BA','TE','AT','TV','ME','CS']

correlations=[]
for col in MA_subdatasets["WL"].columns:  # Transpose to iterate over columns
    correlation = df_MA_corr.AR_Q.corr(MA_subdatasets["WL"][col])
    correlations.append(correlation)
print('The correlation between AR_Q and Water level',correlations)
##############################################################################################

fig, ax = plt.subplots(figsize=(6,3))
ax.set_xlabel('Year')
ax.set_ylabel(r'$\xi$ [m,NAVD88]')

for i,j in enumerate(sort_basin):
    ax.plot(MA_subdatasets["WL"].index, MA_subdatasets["WL"][j],color=color_palette[i],linewidth=1,label=j)

plt.xlim(mp_dates.date2num(plot_period))
plt.ylim([0, 1.0])
major_ticks = np.arange(0, 1.1, 0.5)
#minor_ticks = np.arange(0, 0.31, 1)
ax.set_yticks(major_ticks)
plt.grid(color = 'k', linestyle = '--', linewidth = 0.1)
B,C =ax.get_legend_handles_labels()
ax.legend(C,loc='upper left',ncol=2) # Rabeling
output = os.path.join(Photospace, 'WL_basin.png')
plt.savefig(output,dpi=600,bbox_inches='tight')
#plt.show()
plt.close()

trends= calculate_trends(MA_subdatasets, "WL",subdomain=True, sort_domain_list=sort_basin)
print("\n\n","CRMS trends for climate driver is",trends)

#############################################
# Basin level precipitation
#############################################

# file_name9="Basin_prcp"
file_name9="Basin_total_prcp" # Total precipitation have to update Jin 07/17/24
# file_name="CRMS_Surface_salinity_2006_2022"
file_name9=file_name9+file_suffix2

Basin_prcp = create_dataframe(file_name9, "Date")

# Calculate moving average
MA_datasets_Basin_prcp= Basin_prcp.rolling(window=MA_window, center=True).mean() # , min_periods=9min_period is a rough criterion

#############################################
# Basin level correlation between CRMS and Climate driver
#############################################
SST_basin = SST.copy().query('index >= @start_date and index <= @end_date')
# need to update the SST_basin data to match the basin data
#corr_basin = create_subdomain_correlation(subdatasets, sorted_file_name_o, sort_basin, SST_basin, CS_Q, Basin_prcp, Subbasinspace, 'basin')

#############################################
# Hydroperiod and inundation depth plots
#############################################

# Create 5 years interval data
# Define the start and end years
start_year = int(start_date.split('-')[0])  # start_date
end_year = int(end_date.split('-')[0])  # end_date
interval = 5

sorted_file_sub_name = get_sorted_files(Subbasinspace, 'W_HP_subset_*.csv', "W_HP_subset_basin_")
sorted_file_sub_name2 = get_sorted_files(Subbasinspace, 'W_depth_subset_*.csv', "W_depth_subset_basin_")
sorted_file_sub_name3 = get_sorted_files(Subbasinspace, 'W_HP_SLR_subset_*.csv', "W_HP_SLR_subset_basin_")
sorted_file_sub_name4 = get_sorted_files(Subbasinspace, 'W_depth_SLR_subset_*.csv', "W_depth_SLR_subset_basin_")

datasets_HP = {}
datasets_HP.update(process_files(sorted_file_sub_name, sort_basin, "Now"))
datasets_HP.update(process_files(sorted_file_sub_name3, sort_basin, "SLR"))

print(datasets_HP['Now']['PO']) # check the datasets

HP_stats = pd.DataFrame()
# HP_stats = pd.DataFrame()

for i in ["Now"]:
    aa = [] # dummy list
    for j in sort_basin:
        aa.append(datasets_HP[i][j].median())
    HP_stats[f"HP_{2008}"] = [item.iloc[2] for item in aa]
    HP_stats[f"Depth_{2008}"] = [item.iloc[3] for item in aa]
    HP_stats[f"HP_{2018}"] = [item.iloc[6] for item in aa]
    HP_stats[f"Depth_{2018}"] = [item.iloc[7] for item in aa]

HP_stats.index = np.array(sort_basin).T
HP_stats['HP_diff'] = HP_stats.iloc[:, 2] - HP_stats.iloc[:, 0]
HP_stats['Depth_diff'] = HP_stats.iloc[:, 3] - HP_stats.iloc[:, 1]
HP_stats['HP_ratio'] = HP_stats.iloc[:, 2] / HP_stats.iloc[:, 0]
HP_stats['Depth_ratio'] = HP_stats.iloc[:, 3] / HP_stats.iloc[:, 1]
HP_stats.to_csv('HP_interval10.csv')

# for i in ["Now", "SLR"]:
#     aa = []
#     for j in sort_basin:
#         aa.append(datasets_HP[i][j].median())
#     HP_stats_SLR[f"HP_{i}"] = [item.iloc[0] for item in aa]
#     HP_stats_SLR[f"Depth_{i}"] = [item.iloc[1] for item in aa]
#
# HP_stats_SLR.index=np.array(sort_basin).T
# HP_stats_SLR['HP_diff']=HP_stats_SLR.iloc[:,2]-HP_stats_SLR.iloc[:,0]
# HP_stats_SLR['Depth_diff']=HP_stats_SLR.iloc[:,3]-HP_stats_SLR.iloc[:,1]
# HP_stats_SLR['HP_ratio']=HP_stats_SLR.iloc[:,2]/HP_stats_SLR.iloc[:,0]
# HP_stats_SLR['Depth_ratio']=HP_stats_SLR.iloc[:,3]/HP_stats_SLR.iloc[:,1]
# HP_stats_SLR.to_csv('HP.csv')

print(HP_stats)

#######################################################################################################################
# plot
#######################################################################################################################
nested_data = datasets_HP["Now"]
period = ['Total', '08-12', '13-17', '18-22']
reshaped_data = []

for location, variables in nested_data.items():
    print (location)
    col_list = np.arange(1, len(variables.columns), 2)
    col_nums = col_list.tolist()

    HP = variables.iloc[:, ::2].copy()
    HP_merge = np.array(HP.values.flatten())
    print('Station num', int(HP_merge.size/4))

    Depth = variables.iloc[:, col_nums].copy()
    Depth_merge = np.array(Depth.values.flatten())

    periods = np.array(period * variables.shape[0]*col_list.size)
    locations = np.array([location] * variables.shape[0]*col_list.size)

    # Append each row individually
    for loc, per, hp, depth in zip(locations, periods, HP_merge, Depth_merge):
        reshaped_data.append({'Basin': loc, 'Period': per, 'HP': hp, 'Depth': depth})

df = pd.DataFrame(reshaped_data)
df.to_csv('check.csv')
df['Basin'] = pd.Categorical(df['Basin'], ["CS", "ME", "TV", "AT", "TE", "BA", "MR", "BS", "PO"])

# Create a box plot using seaborn
plt.figure(figsize=(8, 5))
sns.set_theme(style='whitegrid',
              palette='Greys',  #hls
              font_scale=1)
# flierprops = dict(marker='o', markerfacecolor= 'none', markersize=1,
#                   linewidth=0, markeredgecolor='grey')

ax=sns.boxplot(x='Basin', y='HP', hue='Period', data=df, showfliers=False,whis=0,linewidth=.5,medianprops={"linewidth": 2,
                        "solid_capstyle": "butt"})
ax.set_ylabel('$\it{HP}$')
ax.set_ylim([0, 1])  # Set y-axis limits
ax.yaxis.set_major_locator(MultipleLocator(0.2))
# plt.title('Box Plot of HP for Different Periods and Locations')
output = os.path.join(Photospace, 'HP_boxplot.png')
plt.savefig(output,dpi=600,bbox_inches='tight')
#plt.show()
plt.close()

# Create a box plot using seaborn
plt.figure(figsize=(8, 5))
sns.set_theme(style='whitegrid',
              palette='Greys',  #hls
              font_scale=1)
# flierprops = dict(marker='o', markerfacecolor= 'none', markersize=1,
#                   linewidth=0, markeredgecolor='grey')

ax=sns.boxplot(x='Basin', y='Depth', hue='Period', data=df, showfliers=False,whis=0,linewidth=.5,medianprops={"linewidth": 2,
                        "solid_capstyle": "butt"})
ax.set_ylabel('$\it{h}$')
ax.set_ylim([0, 0.5])  # Set y-axis limits
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
# plt.title('Box Plot of HP for Different Periods and Locations')
output = os.path.join(Photospace, 'Depth_boxplot.png')
plt.savefig(output,dpi=600,bbox_inches='tight')

#plt.show()
plt.close()


# reset to the default settings
sns.reset_defaults()


print('##########################################################################################################################\n')
print('Grouped by vegetation')
print('##########################################################################################################################\n')

# subset_file=[Temp_point,Salinity_point,W2l_point,W2m_point]
# subset_file_name=["Temp","Salinity","WL","W2M"]

subsets_vegetation = {} # add basin information on sub dataset
grouped_data_vegetation = {} # grouped data in each subset

for file, name in zip(subset_file, subset_file_name):
    print(name)
    try:
        #subsets_vegetation[name] = gpd.sjoin(file, polygon,how='inner',predicate='within')

        df_merged = pd.merge(file, Basin_community[['Community', 'Count','Rate','BASIN', Basin_community.columns[1]]], how='left',
                     left_on=file.columns[0], right_on= Basin_community.columns[1])
        df_merged = df_merged.loc[~df_merged['Count'].isna()] # remove no basin datasets
        subsets_vegetation[name]=df_merged
        #output_subsets =name + "_vege_subset.csv"
        #subsets_vegetation[name].to_csv(output_subsets)
        grouped_data_vegetation[name] = subsets_vegetation[name].groupby('Community')

    except UnicodeDecodeError:
        # If the above fails due to an encoding error, try another encoding
        print('Encoding error')

vegetation_list=set(subsets_vegetation["Temp"].Community.values)

# Create a index list for vegetation dataset
vegetation_index, vegetation_key = create_subdomain_index(grouped_data_vegetation, vegetation_list)
print(vegetation_index.keys())
print (vegetation_index['Temp'])

sort_community=sorted(vegetation_list)
MA_subsets_vegetation = create_subdomain_datasets(datasets, file_name_o, vegetation_index, sort_community, Submarshspace, 'vegetation')

#############################################
# Make median datasets for each vegetation and variable
#############################################
file1=Submarshspace +'/*_median.*csv'
file_sub_name1=[]

for sub_file in glob.glob(file1):
    if "_SLR_" not in sub_file:
        print(sub_file)
        file_sub_name1.append(sub_file)

file2=Submarshspace +'/*_median_MA.*csv'
file_sub_name2=[]

for sub_file in glob.glob(file2):
    if "_SLR_" not in sub_file:
        print(sub_file)
        file_sub_name2.append(sub_file)

sorted_file_name_o=['Salinity', 'Temp', 'WL', 'W_depth', 'W_HP'] # need to reorder the variables
print(sorted_file_name_o)

subdatasets_vegetation=sub_dataframe_gen(file_sub_name1,sorted_file_name_o)
MA_subdatasets_vegetation=sub_dataframe_gen(file_sub_name2,sorted_file_name_o)

#############################################
# Display correlation plots
#############################################
# corr_vegetation={}
# for i in sorted_file_name_o:
#     print(i)
#     corr_vegetation[i]=subdatasets_vegetation[i].corr()
#
#     output = os.path.join(Photospace,'sub_vegetation'+ i+ '.png')
#     heatplot(corr_vegetation[i], output)
#
# Temp

fig, ax = plt.subplots(figsize=(6,3))
ax.set_xlabel('Year')
ax.set_ylabel(u'Temp [℃]')
lab=[]
for i,j in enumerate(sort_community):
    ax.plot(MA_subdatasets_vegetation["Temp"].index, MA_subdatasets_vegetation["Temp"][j],color=color_palette_vegetation[i],linewidth=1,label=j)
plt.xlim(mp_dates.date2num(plot_period))
plt.ylim([15, 30])
major_ticks = np.arange(15, 31, 5)
ax.set_yticks(major_ticks)
plt.grid(color = 'k', linestyle = '--', linewidth = 0.1)
B,C =ax.get_legend_handles_labels()
ax.legend(C,loc='lower right') # Rabeling

B,C =ax.get_legend_handles_labels()
ax.legend(C,loc='lower right') # Rabeling
output = os.path.join(Photospace, 'Temp_marsh.png')
plt.savefig(output,dpi=600,bbox_inches='tight')
#plt.show()
plt.close()
#
# Salinity
#
fig, ax = plt.subplots(figsize=(6,3))
ax.set_xlabel('Year')
ax.set_ylabel('$\it{S}$ [ppt]')

for i,j in enumerate(sort_community):
    ax.plot(MA_subdatasets_vegetation["Salinity"].index, MA_subdatasets_vegetation["Salinity"][j],color=color_palette_vegetation[i],linewidth=1,label=j)

plt.xlim(mp_dates.date2num(plot_period))
plt.ylim([0, 20])
major_ticks = np.arange(0, 21, 4)
#minor_ticks = np.arange(0, 0.31, 1)
ax.set_yticks(major_ticks)
plt.grid(color = 'k', linestyle = '--', linewidth = 0.1)
B,C =ax.get_legend_handles_labels()
ax.legend(C,loc='upper right',ncol=2) # Rabeling
output = os.path.join(Photospace, 'Salinity_marsh.png')
plt.savefig(output,dpi=600,bbox_inches='tight')
#plt.show()
plt.close()

trends= calculate_trends(MA_subdatasets_vegetation, "Salinity",subdomain=True, sort_domain_list=sort_community)
print("\n\n","CRMS trends for climate driver is",trends)

print (MA_subdatasets_vegetation["Salinity"]["Freshwater"].mean(skipna=True))

# WL
fig, ax = plt.subplots(figsize=(6,3))
ax.set_xlabel('Year')
ax.set_ylabel(r'$\xi$ [m,NAVD88]')

for i,j in enumerate(sort_community):
    ax.plot(MA_subdatasets_vegetation["WL"].index, MA_subdatasets_vegetation["WL"][j],color=color_palette_vegetation[i],linewidth=1,label=j)
plt.xlim(mp_dates.date2num(plot_period))
plt.ylim([0, 1.0])
major_ticks = np.arange(0, 1.1, 0.5)
# minor_ticks = np.arange(0, 0.31, 1)
ax.set_yticks(major_ticks)
plt.grid(color='k', linestyle='--', linewidth=0.1)
B, C = ax.get_legend_handles_labels()
ax.legend(C, loc='upper left', ncol=2)  # labeling
output = os.path.join(Photospace, 'WL_marsh.png')
plt.savefig(output, dpi=600, bbox_inches='tight')
#plt.show()
plt.close()

trends = calculate_trends(MA_subdatasets_vegetation, "WL", subdomain=True, sort_domain_list=sort_community)
print("\n\n", "CRMS trends for climate driver is", trends)


# Create a station precipitation data using basin prep
station_prcp = pd.DataFrame(columns=Basin_community['CRMS_Site'].values,index = Basin_prcp.index)

for i,j in enumerate(station_prcp.columns):
    # print(j)
    community_basin=Basin_community['BASIN'][i]
    if community_basin == 'Pearl': # I don't create Pearl basin
        pass
    else:
        station_prcp[j]=Basin_prcp[community_basin]
print (station_prcp)

# Calculate community prep
community_prcp = pd.DataFrame(columns=sort_community,index = Basin_prcp.index)

grouped=Basin_community.groupby('Community')

# Get the index of each group
#vegetation_index["Salinity"]

# # Print the group indices
for group, indices in vegetation_index["Salinity"].items():
    #print(f"Group '{group}': {indices}")
    community_prcp[group]=station_prcp.loc[:,indices].mean(axis=1,skipna=True) # mean the values of each group
print (community_prcp)

#############################################
# community level correlation between CRMS and Climate driver
#############################################
# Need to update the SST_basin data to match the basin data
#corr_community = create_subdomain_correlation(subdatasets_vegetation, sorted_file_name_o, sort_community, SST_basin, CS_Q, community_prcp, Submarshspace, 'community')

print("Job Finished ʕ •ᴥ•ʔ")

#For paper 2 row * 2 columns
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, figsize=(13, 7))
#fig.subplots_adjust(wspace=0.15, hspace=0.2)
fig.subplots_adjust(wspace=0.20, hspace=0.2)

datasets_plot_lists = [MA_subdatasets_vegetation,MA_subdatasets_vegetation,MA_subdatasets, MA_subdatasets]
color_lists = [color_palette_vegetation,color_palette_vegetation,color_palette, color_palette]
y_variables = ['Salinity','WL','Salinity','WL']
types = [sort_community,sort_community,sort_basin, sort_basin]
legend_locs = ['upper center','upper center','upper center','upper left']
text_list = ['(A-1)', '(B-1)', '(A-2)', '(B-2)']
text_list = ['', '', '', '']

for k, (dataset, color_list,y_variable, type,legend_loc) in enumerate(zip(datasets_plot_lists, color_lists, y_variables, types,legend_locs)):
    dataset_plot = dataset[y_variable]
    ax = axes[k//2, k%2]
    for i,j in enumerate(type):

        ax.plot(dataset_plot.index, dataset_plot[j],color=color_list[i],linewidth=1,label=j)

        if  k//2 == 1:
            ax.set_xlabel('Year')

        if y_variable == 'Salinity':
            ax.set_ylabel('$\it{S}$ [ppt]')
            ax.set_ylim([0, 24])
            major_ticks = np.arange(0, 25, 4)
        elif y_variable == 'Temp':
            ax.set_ylabel(u'Temp [℃]')
        elif y_variable == 'WL':
            ax.set_ylabel(r'$\xi$ [m,NAVD88]')
            ax.set_ylim([0, 1.0])
            major_ticks = np.arange(0, 1.1, 0.5)
        elif y_variable == 'W_depth':
            ax.set_ylabel('$\it{h}$')
        elif y_variable == 'W_HP':
            ax.set_ylabel('$\it{HP}$')
        else:
            pass

    ax.set_xlim(mp_dates.date2num(plot_period))


    ax.set_yticks(major_ticks)
    ax.grid(color='k', linestyle='--', linewidth=0.1)
    B,C =ax.get_legend_handles_labels()
    ax.legend(C,loc=legend_loc ,ncol=2) # Rabeling

    # Add the text from the text_list to the upper left corner of the plot
    if text_list is not None and k < len(text_list):
        ax.text(0.01, 0.98, text_list[k], transform=ax.transAxes, fontsize=12, weight='semibold',
                verticalalignment='top')

output = os.path.join(Photospace, 'Salinity_WL_combine.png')
plt.savefig(output,dpi=300,bbox_inches='tight')
plt.show()
plt.close()

# # For paper 1 row * 2 columns
# sns.set_theme(style="whitegrid")
# fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, figsize=(12,4))
# fig.subplots_adjust(wspace=0.3, hspace=0.3)
#
# datasets_plot_lists = [MA_subdatasets_vegetation,MA_subdatasets_vegetation]
# color_lists = [color_palette_vegetation,color_palette_vegetation]
# y_variables = ['Salinity','WL']
# types = [sort_community,sort_community]
# legend_locs = ['upper right','upper right']
# text_list = ['(A)', '(B)']
#
# for k, (dataset, color_list,y_variable, type,legend_loc) in enumerate(zip(datasets_plot_lists, color_lists, y_variables, types,legend_locs)):
#     dataset_plot = dataset[y_variable]
#     ax = axes[k]
#     for i,j in enumerate(type):
#
#         ax.plot(dataset_plot.index, dataset_plot[j],color=color_list[i],linewidth=1,label=j)
#         ax.set_xlabel('Year')
#
#         if y_variable == 'Salinity':
#             ax.set_ylabel('$\it{S}$ [ppt]')
#             ax.set_ylim([0, 20])
#             major_ticks = np.arange(0, 21, 4)
#         elif y_variable == 'Temp':
#             ax.set_ylabel(u'Temp [℃]')
#         elif y_variable == 'WL':
#             ax.set_ylabel(r'$\xi$ [m,NAVD88]')
#             ax.set_ylim([0, 0.8])
#             major_ticks = np.arange(0, 0.9, 0.4)
#         elif y_variable == 'W_depth':
#             ax.set_ylabel('$\it{h}$')
#         elif y_variable == 'W_HP':
#             ax.set_ylabel('$\it{HP}$')
#         else:
#             pass
#
#     ax.set_xlim(mp_dates.date2num(plot_period))
#
#
#     ax.set_yticks(major_ticks)
#     ax.grid(color='k', linestyle='--', linewidth=0.1)
#     B,C =ax.get_legend_handles_labels()
#     ax.legend(C,loc=legend_loc ,ncol=2) # Rabeling
#
#     # Add the text from the text_list to the upper left corner of the plot
#     if text_list is not None and k < len(text_list):
#         ax.text(0.01, 0.98, text_list[k], transform=ax.transAxes, fontsize=12, weight='semibold',
#                 verticalalignment='top')
#
# output = os.path.join(Photospace, 'Salinity_WL_combine_single.png')
# plt.savefig(output,dpi=600,bbox_inches='tight')
# plt.show()
# plt.close()