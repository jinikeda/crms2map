#!/usr/bin/env python
# coding: utf-8
# CRMS2Regression_analysis plot. Monthly_analysis.py and Bootstrap_Regression_analysis.py created input data
# This file is developed for plotting stepwise linear and random forest regression analyses.
# Developed by the Center for Computation & Technology at Louisiana State University (LSU).
# Developer: Jin Ikeda
# Last modified Dec 6, 2024

### parameters ###################################################################################################
start_date = '2008-01-01' # 'yyyy-mm-dd'
plot_period = [start_date, '2023-01-01'] # 'yyyy-mm-dd'

### Functions ####################################################################################################

# for regression analysis
def process_file(base_path, file, drop_varibales_list=[],nrows=None):
    file_path = os.path.join(base_path, file)
    print(file)
    try:
        df = pd.read_excel(file_path,nrows=nrows)
    except Exception as e:
        print('Error reading file:', e)
        return None
    #print(df.columns)
    if 'Date' in df.columns:
        df = date2datetime(df).copy()
    elif 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'variables'}, inplace=True)
        df = date2datetime(df, column='variables').copy()
    else:
        pass
    df.drop(drop_varibales_list, axis=1, inplace=True)
    return df

def generate_nested_dataframes_plot(basin_list, base_path, drop_varibales_list=[]):
    dataframes1 = {} #time series data
    dataframes2 = {} #stats summary
    for basin in basin_list:
        dataframes1[basin] = {}
        dataframes2[basin] = {}

        file1 = 'model_fit_lm_test_plot_Salinity_' + basin + '.xlsx'
        file2 = 'model_fit_rf_test_plot_Salinity_' + basin + '.xlsx'
        file3 = 'model_fit_lm_test_plot_WL_' + basin + '.xlsx'
        file4 = 'model_fit_rf_test_plot_WL_' + basin + '.xlsx'

        file5 = 'impotance_lm_Salinity_' + basin + '.xlsx'
        file6 = 'impotance_rf_Salinity_' + basin + '.xlsx'
        file7 = 'impotance_lm_WL_' + basin + '.xlsx'
        file8 = 'impotance_rf_WL_' + basin + '.xlsx'

        file1_o = 'lm_Salinity'
        file2_o = 'rf_Salinity'
        file3_o = 'lm_WL'
        file4_o = 'rf_WL'

        file_list1 = [file1, file2, file3, file4]
        file_list2 = [file5, file6, file7, file8]
        file_list_o = [file1_o,file2_o,file3_o,file4_o]

        for file, name_o in zip(file_list1, file_list_o): # time series data
            df = process_file(base_path, file, drop_varibales_list)
            dataframes1[basin][name_o] = df

        for file, name_o in zip(file_list2, file_list_o):  # boxplot data
            df = process_file(base_path, file, drop_varibales_list)
            if 'lm' in file:
                df.drop(['Mean','STD','Q1','Q2','Q3'], axis=1, inplace=True) # drop stats summary because it is not include np.nan values
                df.loc['const'] = np.nan
                df = df.fillna(0)
                ordered_index = ['const', 'GoM SST', 'Q', 'Prcp', 'U10', 'V10']
                df = df.reindex(ordered_index) # reorder index
                df.rename(index={'Prcp': 'P'}, inplace=True) # change the display name here I don't want to change as italic style
                df = add_stats(df, df.columns.get_loc('trial_0'))
            else:
                df.rename(index={'Prcp': 'P'}, inplace=True)
            dataframes2[basin][name_o] = df

    return dataframes1,dataframes2, file_list_o


def generate_nested_summary(basin_list, base_path, drop_varibales_list=[],nrows=None):
    dataframes = {} # summurize result
    radj = {} # r2 adjusted

    for basin in basin_list:
        dataframes[basin] = {}
        radj[basin] = {}

        file1 = 'model_fit_Salinity_' + basin + '.xlsx'
        file2 = 'model_fit_WL_' + basin + '.xlsx'

        file3 = 'model_fit_r2ad_Salinity_'+ basin + '.xlsx'
        file4 = 'model_fit_r2ad_WL_' + basin + '.xlsx'

        file1_o = 'Salinity'
        file2_o = 'WL'

        file_list1 = [file1, file2]
        file_list2 = [file3, file4]
        file_list_o = [file1_o,file2_o]

        for file, name_o in zip(file_list1, file_list_o): # time series data
            df = process_file(base_path, file, drop_varibales_list, nrows)
            df2 = pd.DataFrame() # create new dataframe
            df2['MAE'] = df['MAE'].dropna()
            df2['IA'] = df['IA'].dropna()
            df2.reset_index(drop=True, inplace=True)
            index_labels = ['SL', 'RF']
            df2.index = index_labels
            df2.index.name = 'Model'
            dataframes[basin][name_o] = df2

        for file, name_o in zip(file_list2, file_list_o): # time series data
            df = process_file(base_path, file, drop_varibales_list,nrows=None)
            radj[basin][name_o] = df['R2ad'].mean()

    return dataframes, radj, file_list_o

def date2datetime(df,column='Date'):
    if column == 'Date':
        df.index = pd.to_datetime(df[column])
        df.drop([column], axis=1, inplace=True)
    else:
        df.set_index(column, inplace=True)
    return df

def add_stats(df, first_col): # Caution excel output includes 'xxx. " ' " cannot be treated as numeric and differs from this output.
    # first col=0 for statistical analyssis
    quantiles = df.iloc[:, first_col:].quantile([0.25, 0.5, 0.75], axis=1, numeric_only=True, interpolation='linear')
    df['Mean'] = df.iloc[:, first_col:].mean(axis=1, skipna=True)
    df['STD'] = df.iloc[:, first_col:].std(axis=1, skipna=True)
    df['Q1'] = quantiles.loc[0.25]
    df['Q2'] = quantiles.loc[0.5]
    df['Q3'] = quantiles.loc[0.75]

    return df

def model_plot_test(dataframes,basin,name_o,plot_period):
    plt.clf()

    # create plot
    fig, ax = plt.subplots(figsize=(6, 3))

    ax.set_xlabel('Year')
    ax.set_ylabel('$\it{S}$ [ppt]')
    lab5, = ax.plot(dataframes[basin][name_o[0]].index, dataframes[basin][name_o[0]]["Obs"], 'k-', linewidth=1.5, label='Obs')
    lab6, = ax.plot(dataframes[basin][name_o[0]].index, dataframes[basin][name_o[0]]["Q2"], 'b--', linewidth=1, label='SL')
    lab7, = ax.plot(dataframes[basin][name_o[1]].index, dataframes[basin][name_o[1]]["Q2"], 'r--', linewidth=1, label='RF')

    plt.xlim(mp_dates.date2num(plot_period))

    if basin == 'CS':
        plt.ylim([0, 30])
        major_ticks = np.arange(0, 31, 6)
    else:
        plt.ylim([0, 18])
        major_ticks = np.arange(0, 19, 6)

    # minor_ticks = np.arange(0, 0.31, 1)
    ax.set_yticks(major_ticks)
    plt.grid(color='k', linestyle='--', linewidth=0.1)
    ax.legend(handles=[lab5, lab6, lab7], loc='upper left')
    output = os.path.join(Photospace,'Salinity_fit_test_' + basin + '.png')
    plt.savefig(output, dpi=600, bbox_inches='tight')
    #plt.show()
    plt.close()

    plt.clf()

    # create waterlevel plot
    fig, ax = plt.subplots(figsize=(6, 3))
    # ax.errorbar(CRMS_mean.index.year, CRMS_mean, yerr=CRMS_std, fmt='o', mfc='none',
    # mec='k', ecolor='lightgray',capsize=2,elinewidth=1)
    # ax.plot(datasets["W_HP"].index, datasets["W_HP"].median(axis=1),'o', mfc='none', mec='k')

    ax.set_xlabel('Year')
    ax.set_ylabel(r'$\xi$ [NAVD88,m]')
    lab5, = ax.plot(dataframes[basin][name_o[2]].index, dataframes[basin][name_o[2]]["Obs"], 'k-', linewidth=1.5, label='Obs')
    lab6, = ax.plot(dataframes[basin][name_o[2]].index, dataframes[basin][name_o[2]]["Q2"], 'b--', linewidth=1, label='SL')
    lab7, = ax.plot(dataframes[basin][name_o[3]].index, dataframes[basin][name_o[3]]["Q2"], 'r--', linewidth=1, label='RF')

    plt.xlim(mp_dates.date2num(plot_period))
    if basin == 'AT':
        plt.ylim([-0.2, 1.4])
        major_ticks = np.arange(-0.2, 1.5, 0.4)
    else:
        plt.ylim([-0.2, 0.8])
        major_ticks = np.arange(-0.2, 0.9, 0.2)

    # minor_ticks = np.arange(0, 0.31, 1)
    ax.set_yticks(major_ticks)
    # ax.set_yticks(minor_ticks)
    plt.grid(color='k', linestyle='--', linewidth=0.1)
    ax.legend(handles=[lab5, lab6, lab7], loc='upper left')
    output = os.path.join(Photospace,'Water_level_fit_test_' + basin + '.png')
    plt.savefig(output, dpi=600, bbox_inches='tight')
    #plt.show()
    plt.close()

    print('Finish Plots')
    return


# exclude_stats
def drop_transpose_stats(df):
    df_processed = df.copy()
    df_processed.drop(['const'], axis=0, inplace=True) # drop index 'const'
    df_processed.drop(['Mean','STD','Q1','Q2','Q3'], axis=1, inplace=True) # drop Mean,STD,Q1,Q2,Q3
    df_T = df_processed.T.reset_index(drop=True)
    df_T.index.rename('trial', inplace='True')
    return df_T


def generate_summary_table(summary_df,radj,basin,variable):
    flattened_Stats = np.concatenate([summary_df[basin][var].values.flatten() for var in reversed(variable)])  # flatten the MAE and IA values and WL is firster than salnity
    flattened_Radj = np.array([radj[basin][var] for var in reversed(variable)])  # flatten the R2 adjusted values
    flattened_values = np.concatenate([flattened_Stats, flattened_Radj])  # concatenate the flattened MAE, IA, and R2 adjusted values
    transpose_df = pd.DataFrame(flattened_values,columns=[basin]).T # transpose the dataframe
    return transpose_df

def melted_table(df1, df2):
    lm_table = drop_transpose_stats(df1)
    rf_table = drop_transpose_stats(df2)
    lm_table['Model'] = 'SL'
    rf_table['Model'] = 'RF'
    df = pd.concat([lm_table, rf_table])
    df = df.fillna(-9999)

    melted_df = pd.melt(df, id_vars='Model', value_vars=df.columns[0:6], var_name='Variables',
                        value_name='Relative Importance')

    return melted_df

def boxplot(dataframes,basin,name_o):

    matplotlib.pyplot.text
    matplotlib.axes.Axes.text
    rcParams['font.family'] = 'sans-serif'

    # Plotting the bars
    plt.clf()
    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=False, figsize=(12, 3.5))
    ax1 = axs[1]  # swith salinity and water level
    ax2 = axs[0]

    # Data Salinity
    melted_df = melted_table(dataframes[basin][name_o[0]], dataframes[basin][name_o[1]])
    flierprops = dict(marker='o', markerfacecolor='none', markersize=2,markeredgecolor='grey')
    sns.boxplot(x='Variables', y='Relative Importance', hue='Model', data=melted_df, palette='husl', ax=ax1,flierprops=flierprops)
    ax1.set_ylabel("Relative importance", fontsize=14)
    ax1.set_ylim([0, 1])
    ax1.yaxis.set_major_locator(MultipleLocator(0.25))
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(which='major', color='grey', linestyle='dashed', alpha=0.25)
    ax1.yaxis.grid(which='minor', color='grey', linestyle=':', alpha=0.25)
    ax1.set(xlabel=None)

    # Data water level
    melted_df2 = melted_table(dataframes[basin][name_o[2]], dataframes[basin][name_o[3]])
    flierprops = dict(marker='o', markerfacecolor='none', markersize=2,markeredgecolor='grey')
    sns.boxplot(x='Variables', y='Relative Importance', hue='Model', data=melted_df2, palette='husl', ax=ax2,flierprops=flierprops)
    ax2.set_ylabel("Relative importance", fontsize=14)
    ax2.set_ylim([0, 1])
    ax2.yaxis.set_major_locator(MultipleLocator(0.25))
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(which='major', color='grey', linestyle='dashed', alpha=0.25)
    ax2.yaxis.grid(which='minor', color='grey', linestyle=':', alpha=0.25)
    ax2.set(xlabel=None)

    plt.tight_layout()
    # plt.show()
    output = os.path.join(Photospace,'boxplot_' + basin + '.png')
    fig.savefig(output, dpi=600)
    plt.close()

    print('Finish Plots')
    return


########################################################################
print ('Data_plot')
######################################################################

### 1.1 Import modules ###
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import glob
# stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import statsmodels.api as sm
from scipy import stats
from scipy import signal
from scipy import special
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# plot
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.dates as mp_dates
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from matplotlib import rcParams

Workspace=os.getcwd()
# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))

# Change the current working directory
os.chdir(Workspace)

Photospace = os.path.join(Workspace, 'Photo')
Inputspace = os.path.join(Workspace, 'Sub_basin')
Outputspace=os.path.join(Workspace,'bootstrap_Output')


try:
    #os.makedirs(Inputspace, exist_ok=True)
    #os.makedirs(Photospace, exist_ok=True)
    os.makedirs(Outputspace, exist_ok=True)

except Exception as e:
    print(f"An error occurred while creating directories: {e}")

### Parameters ###########################################################
start_date = '2008-01-01' # 'yyyy-mm-dd'
end_date = '2022-12-31'   # 'yyyy-mm-dd'
threshold1 = 200 # Delete columns where 'num_station' is lower than the threshold for continuous data
threshold2 = int(threshold1/2) # Delete columns where 'num_station' is lower than the threshold for discrete data

plot_period = [start_date, '2023-01-01'] # 'yyyy-mm-dd'

### Step 2 ###########################################################
print ('Step 2: Read input data ')
######################################################################

### 2.1 Read CRMS files ###

file_suffix=".csv"
file_suffix2=".xlsx"

basin_list=['PO','BS','MR','BA','TE','AT','TV','ME','CS']
#basin_list=['AT','TE','CS']

# Make a nested dictionary to store the dataframes
test_plot,test_box,name_o= generate_nested_dataframes_plot(basin_list, Outputspace,drop_varibales_list=[])
summary_df,radj,variable = generate_nested_summary(basin_list, Outputspace,drop_varibales_list=[],nrows=2) # nrows=2 only read first 2 rows


summary_table = pd.DataFrame()
for basin in basin_list:
    print ('Plot_basin', basin)

    model_plot_test(test_plot,basin,name_o,plot_period)
    boxplot(test_box, basin, name_o)

    # make summary table
    transpose_df = generate_summary_table(summary_df,radj,basin,variable)
    summary_table = pd.concat([summary_table, transpose_df])

# # Reset the index and rename the columns
summary_table.columns = ['MAE_WL_SL', 'IA_WL_SL', 'MAE_WL_RF', 'IA_WL_RF', 'MAE_S_SL', 'IA_S_SL', 'MAE_S_RF', 'IA_S_RF','R2ad_WL_SL', 'R2ad_S_SL']
summary_table.index.name = 'Basin'
reorder_list = ['R2ad_WL_SL','MAE_WL_SL', 'IA_WL_SL', 'MAE_WL_RF', 'IA_WL_RF', 'R2ad_S_SL', 'MAE_S_SL', 'IA_S_SL','MAE_S_RF', 'IA_S_RF']
summary_table = summary_table.reindex(reorder_list, axis=1)
output = os.path.join(Outputspace, 'Summary.xlsx')
summary_table.to_excel(output)
print(summary_table)

print("Job Finished ʕ •ᴥ•ʔ")

