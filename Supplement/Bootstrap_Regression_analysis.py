#!/usr/bin/env python
# coding: utf-8
# machine learning for stepwise linear regression and random forest regression
# Developed by the Center for Computation & Technology and Center for Coastal Design Studio at Louisiana State University (LSU).
# Developer: Jin Ikeda
# Last modified July 24, 2024

### Step 3 ###########################################################
print ('Step 3: Fit model and Test')
######################################################################

### 3.1 Import modules ###
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from scipy import special
from scipy import signal
import time
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


##################################################################################################################
### parameters ###################################################################################################

Output_name = "bootstrap_Output"

# Randomly pick up dataset
n_repeat = int(50) #
test_ratio= 0.25

#basin list
#basin_list=['CS']
#basin_list=['PO','BS','MR','ME','CS']
basin_list=['PO','BS','MR','BA','TE','AT','TV','ME','CS']

Workspace=os.getcwd()
# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))

### HPC ####
#Workspace = "/work/jinikeda/ETC/CRMS2Plot/" # if you want to specify the location

# Change the current working directory
os.chdir(Workspace)

# Make Input and output folders
Inputspace = os.path.join(Workspace, 'Sub_basin')
Photospace = os.path.join(Workspace, 'Photo')
Outputspace = os.path.join(Workspace,Output_name)

try:
    #os.makedirs(Inputspace, exist_ok=True)
    os.makedirs(Photospace, exist_ok=True)
    os.makedirs(Outputspace, exist_ok=True)

except Exception as e:
    print(f"An error occurred while creating directories: {e}")

### Function #####################################################################################################

# for regression analysis
def generate_nested_dataframes(basin_list, base_path,drop_varibales_list): # drop_varibales_list = ['W_HP','W_depth'] because these data missing until 2008/03
    dataframes = {}
    for basin in basin_list:
        file_path = os.path.join(base_path, f'basin_{basin}.csv')
        df = pd.read_csv(file_path)
        df = date2datetime(df)
        df.drop(drop_varibales_list, axis=1, inplace=True)
        df.dropna(axis=0, how='any', inplace=True)
        dataframes[basin] = df
    return dataframes


def date2datetime(df,column='Date'):
    if column == 'Date':
        df.index = pd.to_datetime(df[column])
        df.drop([column], axis=1, inplace=True)
    else:
        df.set_index(column, inplace=True)
        df.drop([column], axis=1, inplace=True)
    return df


def normalized(df):
    normalized_df= (df - df.min()) / (df.max() - df.min())+0.0001
    print(normalized_df.describe())
    return normalized_df


# Investigate cross correlation with lags
def ccf_values(series1, series2):
    p = series1
    q = series2
    p = (p - np.mean(p)) / (np.std(p) * len(p))
    q = (q - np.mean(q)) / (np.std(q))
    c = np.correlate(p, q, 'full')
    return c


def ccf_lags(df,y_list): # y_list = ['Salinity', 'WL']

    df2 = pd.DataFrame()

    for y in y_list:
        for x in df.columns[0:5]: # GoM_SST,AR_Q,Prcp,U10,V10
            corr = ccf_values(df[y], df[x])
            lags = signal.correlation_lags(len(df[y]), len(df[x]))
            df2['lags'] = lags
            df2[f'r_{x}_{y}'] = corr

    mid = np.int64(df2.shape[0] / 2)  # center index
    df3 = df2.iloc[mid - 6 : mid + 6+1]
    max_indices = df3.idxmax()
    min_indices = df3.idxmin()

    if len(basin) == 0:
        output = 'cross_correlation.xlsx'
    else:
            output = 'cross_correlation_' + basin + '.xlsx'

    df2.to_excel(os.path.join(Inputspace,output))
    lag_info = []
    for i in min_indices[1:int(df2.shape[1] / 2) + 1]:  # for salinity
        lag_info.append(df2.lags[i])  # Go to first row

    for i in max_indices[int(df2.shape[1] / 2) + 1::]:  # for water level
        lag_info.append(df2.lags[i])  # Go to second row
    lag = np.reshape(lag_info, (2, -1))  # Change the array shape to 2 rows

    print('Lag: ', lag)
    return df2, lag


def modeldataset_shift(df, shift_info):
    # Shift 'Q' and 'Prcp' columns
    df['Q'] = df['Q'].shift(shift_info[0])
    df['Prcp'] = df['Prcp'].shift(shift_info[1])

    df.dropna(axis=0, how='any',inplace=True)
    X, X_significant, y, df = modeldataset(df)

    return X, X_significant, y, df


def collinearity_VIF(df,vif_coef=10): # vif_coef: remove criterion of a variance inflation factor (VIF) value

    df_vif=df.copy()
    while True:
        vif_data = pd.DataFrame()
        vif_data["feature"] = df_vif.columns
        vif_data["VIF"] = [variance_inflation_factor(df_vif.values, i) for i in range(len(df_vif.columns))]
        print(vif_data)
        max_vif = vif_data["VIF"].max()
        if max_vif > vif_coef:
            max_vif_feature = vif_data[vif_data["VIF"] == max_vif]["feature"].values[0]
            df_vif = df_vif.drop(columns=max_vif_feature)
            print('high collinearity!!! Drop a variable of ', max_vif_feature)
        else:
            break
    return df_vif


def modeldataset(df,alpha= 0.05):
    # Normalize independent variables
    X_pre = df.iloc[:, 0:5]
    normalized_df = normalized(X_pre)

    # Apply Box-Cox transformation
    X = pd.DataFrame()
    X.index =df.copy().index
    lmda = pd.DataFrame()

    for column in normalized_df.columns:
        X[column], lmda[column] = stats.boxcox(normalized_df[column].values)
    print(X.describe())

    # for i in X.columns:
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(211)
    #     prob = stats.probplot(X_pre[i], dist=stats.norm, plot=ax1)
    #     ax1.set_xlabel('')
    #     ax1.set_title('Probplot against normal distribution')
    #     ax2 = fig.add_subplot(212)
    #     prob = stats.probplot(X[i], dist=stats.norm, plot=ax2)
    #     ax2.set_title('Probplot after Box-Cox transformation')
    #     plt.show()

    for i in X.columns:
        k2, p = stats.shapiro(X[i])
        print('\nShapiro-Wilk test statistic = %.3f, p = %.3f' % (k2, p))

        if p > alpha:
            print('\nThe transformed data is Gaussian (fails to reject the null hypothesis)')
        else:
            print('\nThe transformed data does not look Gaussian (reject the null hypothesis)')

    # Test for multicollinearity and drop high collinearity_VIF
    df_vif = collinearity_VIF(X)

    # Add constant to independent variables
    X = sm.add_constant(X)
    X_significant = sm.add_constant(df_vif)

    y = df.iloc[:, 6:]

    return X, X_significant, y, df


def stepwise_part (y, X_vif, alpha):
    while True:
        lm_model = sm.OLS(y, X_vif)

        # Fit the model
        results = lm_model.fit()
        p_values = results.pvalues
        p_values_max = p_values.max()
        drop_variable = p_values.idxmax()

        if drop_variable != 'const':  # don't drop const
            if p_values_max > alpha:
                X_vif = X_vif.drop([drop_variable], axis=1) # update the file
            else:
                break
        else:
            # Get the largest p-value excluding 'const'
            second_largest_p_value = p_values[p_values.index != 'const'].max()
            if second_largest_p_value > alpha:
                second_drop_variable = p_values[p_values == second_largest_p_value].index[0]
                X_vif = X_vif.drop([second_drop_variable], axis=1)  # update the file
            else:
                break

    X_significant = X_vif.copy()
    Variable_importance = (results.params.drop('const') ** 2) / (results.params.drop('const') ** 2).sum()

    # print(results.summary())
    # print(Variable_importance)

    return results, X_significant,Variable_importance


# this function is crude codes and needs to be improved by Jin 02/01/2024
def stepwise_model_fit_test_bootstrap(X, X_vif,y,  y_variable, basin,n_repeat, test_ratio, alpha=0.05):

    dropped_vif_variables = list(set(X.columns) - set(X_vif.columns)) # for OLS model drop high collinearity varibales #dropped_vif_variables = [] if no drops

    # For predictive variables
    features = X.copy()
    labels = y.copy()
    y_whole = labels[y_variable].copy()

    ee = pd.DataFrame() #'model_fit_lm_plot_' + y_variable + '_' + basin + '.xlsx'
    dd = pd.DataFrame() #'impotance_rf_' + y_variable + '_' + basin + '.xlsx'
    ff = pd.DataFrame() #'model_fit_rf_plot_' + y_variable + '_' + basin + '.xlsx'
    gg = pd.DataFrame() #'model_fit_r2ad_' + y_variable + '_' + basin + '.xlsx'
    ii = pd.DataFrame() #'impotance_lm_' + y_variable + '_' + basin + '.xlsx'
    jj = pd.DataFrame() #'model_fit_lm_test_plot_' + y_variable + '_' + basin + '.xlsx'
    kk = pd.DataFrame() #'model_fit_rf_test_plot_' + y_variable + '_' + basin + '.xlsx'
    error_table = pd.DataFrame()
    time_index = pd.DataFrame()
    time_index.index = y.index.copy()
    referenced_month = time_index.index[0]  # origin month

    datasets_summary = []

    # Split the data into training and testing sets
    for i in range(n_repeat):
        print(i)
        vif_features = X_vif.copy()
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                                    test_size=test_ratio)
        y_train = train_labels[y_variable]
        y_test = test_labels[y_variable]

        # For linear regression only
        train_features_vif_OLS = train_features.copy()
        train_features_vif_OLS.drop(dropped_vif_variables, axis=1, inplace=True)


        if y_variable == 'Salinity':
            y_boxcox = stats.boxcox(y_train)

            # stepwise linear regression part (drop variable only until p < alpha (keep intercept)
            results, X_significant, Variable_importance = stepwise_part(y_boxcox[0], train_features_vif_OLS, alpha)

            gg.loc[i, 'R2ad'] = results.rsquared_adj

            dropped_significant_variables = list(
                set(train_features_vif_OLS.columns) - set(X_significant.columns))  # for OLS model after drop #dropped_variables = [] if no drops

            #print (dropped_significant_variables)
            vif_features.drop(dropped_significant_variables, axis=1, inplace=True)

            test_features_significant = test_features.copy()
            test_features_significant.drop(dropped_vif_variables, axis=1, inplace=True)
            test_features_significant.drop(dropped_significant_variables, axis=1, inplace=True)

            lm_predict = results.predict(test_features_significant)
            lm_predict_whole = results.predict(vif_features)
            lm_predict = special.inv_boxcox(lm_predict, y_boxcox[1])  # inverse box-cox transform
            lm_predict_whole = special.inv_boxcox(lm_predict_whole, y_boxcox[1]) # inverse box-cox transform

        else:
            # stepwise linear regression part (drop variable only until p < alpha (keep intercept)
            results, X_significant, Variable_importance = stepwise_part(y_train, train_features_vif_OLS, alpha)

            gg.loc[i, 'R2ad'] = results.rsquared_adj

            dropped_significant_variables = list(
                set(train_features_vif_OLS.columns) - set(X_significant.columns))  # for OLS model after drop #dropped_variables = [] if no drops

            #print (dropped_significant_variables)
            vif_features.drop(dropped_significant_variables, axis=1, inplace=True)

            test_features_significant = test_features.copy()
            test_features_significant.drop(dropped_vif_variables, axis=1, inplace=True)
            test_features_significant.drop(dropped_significant_variables, axis=1, inplace=True)

            lm_predict = results.predict(test_features_significant)
            lm_predict_whole = results.predict(vif_features)

        lm_predict.index=y_test.index # for error analysis *lm_predict.index is integer, but y_test.index is datetime
        lm_predict_whole.index=y_whole.index # for error analysis

        # Calculate the absolute errors
        lm_errors = y_test - lm_predict
        #print ('y_test:',y_whole, y_whole.shape)
        lm_errors_whole = y_whole - lm_predict_whole
        print('Mean Absolute Error:', round(np.mean(abs(lm_errors)), 2), 'ppt')


        # Random forest regression
        # Instantiate model with 100 decision trees
        rf = RandomForestRegressor(n_estimators=100, random_state=1)
        # Train the model on training data
        rf.fit(train_features, y_train)

        # Use the forest's predict method on the test data
        rf_predict = rf.predict(test_features)
        rf_predict_whole = rf.predict(features)
        # Calculate the absolute errors
        rf_errors = y_test - rf_predict
        #rf_errors_whole = y_whole - rf_predict_whole
        print('Mean Absolute Error:', round(np.mean(abs(rf_errors)), 2), 'ppt')


        # Get numerical feature importances
        importances = list(rf.feature_importances_)
        # List of tuples with variable and importance
        feature_list = features.columns
        feature_importances = [(feature, round(importance, 2)) for feature, importance in
                               zip(feature_list, importances)]

        if i == 0:

            # Error table
            aa = pd.DataFrame()  # error_table
            aa['Obs'] = y_test
            aa['Pred'] = lm_predict
            aa['Error'] = lm_errors
            aa['RF_Pred'] = rf_predict
            aa['RF_Error'] = rf_errors

            error_table = aa.copy()

            # Importance table
            dd.index = feature_list
            numeric = [pair[1] for pair in feature_importances]
            column_name = 'trial_' + str(i)
            dd[column_name] = numeric

            ii.index = X.copy().columns.drop(['const'])
            ii[column_name] = Variable_importance # for OLS model after drop

            # Sort the feature importances by most important first
            feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
            # Print out the feature and importances
            # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

            # Predictive plot
            column_name2 = 'lm_pr' + str(i)
            column_name3 = 'rf_pr' + str(i)

            ee['Obs'] = y_whole
            ee[column_name2] = lm_predict_whole

            ff['Obs'] = y_whole
            ff[column_name3] = rf_predict_whole

            jj['Obs'] = y_whole
            jj[column_name2] = aa['Pred']

            kk['Obs'] = y_whole
            kk[column_name3] = aa['RF_Pred']

        else:
            # Error table
            bb = pd.DataFrame()
            bb['Obs'] = y_test
            bb['Pred'] = lm_predict
            bb['Error'] = lm_errors
            bb['RF_Pred'] = rf_predict
            bb['RF_Error'] = rf_errors

            error_table = pd.concat([error_table, bb.copy()])

            # Importance table
            numeric = [pair[1] for pair in feature_importances]
            column_name = 'trial_' + str(i)
            dd[column_name] = numeric

            ii[column_name] = Variable_importance # for ols after drop

            # Sort the feature importances by most important first
            feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
            # Print out the feature and importances
            # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

            # Predictive plot
            column_name2 = 'lm_pr' + str(i)
            column_name3 = 'rf_pr' + str(i)

            ee[column_name2] = lm_predict_whole
            ff[column_name3] = rf_predict_whole
            jj[column_name2] = bb['Pred']
            kk[column_name3] = bb['RF_Pred']

    # Add error indeces
    error_table = error_analysis(error_table)
    output = 'model_fit_' + y_variable + '_' + basin + '.xlsx'
    output_dir = os.path.join(Outputspace, output)
    error_table.to_excel(output_dir, index=False, header=True)

    dd = add_stats(dd, 0)
    output2 = 'impotance_rf_' + y_variable + '_' + basin + '.xlsx'
    output_dir = os.path.join(Outputspace, output2)
    dd.to_excel(output_dir, index=True, header=True)

    ii.fillna(0) # include no contributions
    ii = add_stats(ii, 0)
    output2 = 'impotance_lm_' + y_variable + '_' + basin + '.xlsx'
    output_dir = os.path.join(Outputspace, output2)
    ii.to_excel(output_dir, index=True, header=True)

    ee.reset_index(drop=True)
    ee.index = time_index.index
    ee = add_stats(ee, 1)
    ff.reset_index(drop=True)
    ff.index = time_index.index
    ff = add_stats(ff, 1)
    jj.reset_index(drop=True)
    jj.index = time_index.index
    jj = add_stats(jj, 1)
    kk.reset_index(drop=True)
    kk.index = time_index.index
    kk = add_stats(kk, 1)

    output3 = 'model_fit_lm_plot_' + y_variable + '_' + basin + '.xlsx'
    output_dir = os.path.join(Outputspace, output3)
    ee.to_excel(output_dir, index=True, header=True)

    output5 = 'model_fit_r2ad_' + y_variable + '_' + basin + '.xlsx'
    output_dir = os.path.join(Outputspace, output5)
    gg.to_excel(output_dir, index=True, header=True)

    output4 = 'model_fit_rf_plot_' + y_variable + '_' + basin + '.xlsx'
    output_dir = os.path.join(Outputspace, output4)
    ff.to_excel(output_dir, index=True, header=True)

    output = 'model_fit_lm_test_plot_' + y_variable + '_' + basin + '.xlsx'
    output_dir = os.path.join(Outputspace, output)
    jj.to_excel(output_dir, index=True, header=True)

    output = 'model_fit_rf_test_plot_' + y_variable + '_' + basin + '.xlsx'
    output_dir = os.path.join(Outputspace, output)
    kk.to_excel(output_dir, index=True, header=True)

    return lm_predict, rf_predict, error_table

def error_analysis(df):
    df = df.reset_index()
    refdata = df.iloc[:, 1]  # observation
    model = df.iloc[:, 2]  # lm_prediction
    data_frame = df.iloc[:, 3]  # lm_errors
    S = data_frame.dropna().shape[0]  # Number of data
    MEAN = data_frame.mean(axis=0, skipna=True)  # Bias is -1*MEAN
    STD = data_frame.std(axis=0, skipna=True, ddof=0)  # Population standard deviation
    SE = data_frame.sem(axis=0, skipna=True, ddof=0)  # Mean Standard Error = std/(n**0.5)
    MAE = np.mean(abs(data_frame))
    MSE = data_frame.div(STD).mean(axis=0, skipna=True)  # Mean Standardized Error = (error/std)/n
    RMSE = (data_frame ** 2).mean(axis=0, skipna=True) ** 0.5  # Root square mean error
    RMSSE = ((data_frame.div(STD)) ** 2).mean(axis=0,
                                              skipna=True) ** 0.5  # Root square mean error #Root Mean Square Standardized Error
    MAPE = 100 - np.mean(100 * abs(MAE/ df.iloc[:, 1]))

    se = (data_frame ** 2)
    mo = np.sqrt(np.nanmean((refdata) ** 2))  # root mean square of reference data
    SI = STD / mo  # Scatter index
    O = np.nanmean(np.nanmean(model))
    sO = model - O
    P = np.nanmean(np.nanmean(refdata))
    sP = refdata - P
    IA = 1 - np.sum(se) / (np.sum(sO ** 2) + np.sum(sP ** 2) + S * (O - P) ** 2 + 2 * abs(np.sum(sO * sP)))

    model2 = df.iloc[:, 4]  # rf_prediction
    data_frame2 = df.iloc[:, 5]  # rf_errors
    S2 = data_frame2.dropna().shape[0]  # Number of data
    MEAN2 = data_frame2.mean(axis=0, skipna=True)  # Bias is -1*MEAN
    STD2 = data_frame2.std(axis=0, skipna=True, ddof=0)  # Population standard deviation
    SE2 = data_frame2.sem(axis=0, skipna=True, ddof=0)  # Mean Standard Error = std/(n**0.5)
    MAE2 = np.mean(abs(data_frame2))
    MSE2 = data_frame2.div(STD).mean(axis=0, skipna=True)  # Mean Standardized Error = (error/std)/n
    RMSE2 = (data_frame2 ** 2).mean(axis=0, skipna=True) ** 0.5  # Root square mean error
    RMSSE2 = ((data_frame2.div(STD)) ** 2).mean(axis=0,
                                                skipna=True) ** 0.5  # Root square mean error #Root Mean Square Standardized Error
    MAPE2 = 100 - np.mean(100 * abs(MAE2 / df.iloc[:, 1]))

    se2 = (data_frame2 ** 2)
    mo2 = np.sqrt(np.nanmean((refdata) ** 2))  # root mean square of reference data
    SI2 = STD2 / mo2  # Scatter index
    O2 = np.nanmean(np.nanmean(model2))
    sO2 = model2 - O2
    P2 = np.nanmean(np.nanmean(refdata))
    sP2 = refdata - P2
    IA2 = 1 - np.sum(se2) / (np.sum(sO2 ** 2) + np.sum(sP2 ** 2) + S2 * (O2 - P2) ** 2 + 2 * abs(np.sum(sO2 * sP2)))

    df.loc[0, 'num_sample'] = S
    df.loc[0, 'MAE'] = MAE
    df.loc[0, 'MAPE'] = MAPE
    df.loc[0, 'SI'] = SI
    df.loc[0, 'IA'] = IA
    df.loc[0, 'MSE'] = MSE
    df.loc[0, 'RMSE'] = RMSE
    df.loc[0, 'RMSSE'] = RMSSE
    df.loc[0, 'mean_e'] = MEAN
    df.loc[0, 'std_e'] = STD
    df.loc[1, 'num_sample'] = data_frame2.count()
    df.loc[1, 'MAE'] = MAE2
    df.loc[1, 'MAPE'] = MAPE2
    df.loc[1, 'SI'] = SI2
    df.loc[1, 'IA'] = IA2
    df.loc[1, 'MSE'] = MSE2
    df.loc[1, 'RMSE'] = RMSE2
    df.loc[1, 'RMSSE'] = RMSSE2
    df.loc[1, 'mean_e'] = MEAN2
    df.loc[1, 'std_e'] = STD2

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

### Step 2 ###########################################################
print ('Step 2: Read input data ')
######################################################################

### 2.1 Read CRMS files ###

file_suffix=".csv"
file_suffix2=".xlsx"

#basin_list=['PO','BS','MR','BA','TE','AT','TV','ME','CS']

Basin_df = generate_nested_dataframes(basin_list, Inputspace,['W_HP','W_depth']) # drop_variable_list=['W_HP','W_depth']
#print (Basin_df['PO'].columns)

### Step 3 ###########################################################
print ('Step 3: Get output data ')
######################################################################

# #Get lags information
lag_info = {}
y_list = ['Salinity', 'WL'] #['Salinity', 'WL']
for basin in basin_list:
    print(
        '##########################################################################################################################\n')
    print('Basin is ', basin)
    print(
        '##########################################################################################################################\n')
    corr,lag_info[basin]=ccf_lags(Basin_df[basin],y_list) # y_list = ['Salinity', 'WL']

    Basin=Basin_df[basin]
    lag_info[basin][lag_info[basin] < 0] = 0 # physically Q and precipitation proceeds to salinity but no behinds.
    shift_info=[lag_info[basin][0][1],lag_info[basin][0][2]]

    print ('y_list', y_list[0],'df_shape',Basin.shape, 'shift_info [Q,P]',shift_info)
    X, X_vif, y, df_shift = modeldataset_shift(Basin.copy(),shift_info)  # for salinity # causion modeldataset_shift overwrite Basin. So, use Basin.copy()
    stepwise_model_fit_test_bootstrap(X, X_vif, y,y_list[0],basin, n_repeat, test_ratio)

    print('y_list', y_list[1], 'df_shape', Basin.shape)
    X, X_vif, y, dummy_df_shift = modeldataset(Basin.copy())  # for water level
    stepwise_model_fit_test_bootstrap(X, X_vif, y,y_list[1],basin, n_repeat, test_ratio)

print("Time to Compute: \t\t\t", time.process_time(), " seconds")
print("Job Finished ʕ •ᴥ•ʔ")

