# CRMS2Map
Data analytical and mapping tools for the Louisiana Coastal Reference Monitoring System (CRMS) Hydrographic Data from https://cims.coastal.la.gov/monitoring-data/

## §1. Install virtual conda env and activation
Type: ***conda env create -f env.yml*** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; black is an uncompromising Python code formatter, flack8 is a linter to check code, and pytest and mock are code testing tools. These packages are not mandatory for running CRMS2Map  

## §2. Activate virtual env
Type: ***conda activate CRMS2Map***

## §3 Create a package
Type: ***pip install -e .*** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -e: editable mode (Preferred for package developers)

## Contents of the package
### Step 1. Auto retrieve the original datasets

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Hydrographic Hourly Data: Hourly hydrographic data (over 60GB) -> CRMS_Continuous_Hydrographic2subsets.py \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <mark>**[Caution] Need large memory for processing**</mark> \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Hydrographic Monthly Data: Monthly hydrographic data (Manually corrected such as pore water salinity) -> CRMS_Discrete_Hydrographic2subsets.py

### Step 2. Resample the retived data

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; From the step 1, resample hourly, daily, monthly and yearly -> CRMS2Resample.py
### Step 3. Plot time series data (optional)    

## Running a package
* ***CRMS2Map_continuous*** for step. 1: CRMS_Continuous_Hydrographic2subsets.py
* ***CRMS2Map_discrete*** for step.1: Create a subset data for salinity, water temperature, water level, and water level to marsh (hydroperiod and inundation depth) using CRMS_Discrete_Hydrographic2subsets.py
* ***CRMS2Map_resample*** for step.2: Create hourly, daily, monthly, and yearly averaged datasets using CRMS2Resample.py
* ***CRMS2Plot***: Data analysis and plotting tool after step.2\
  This tool creates a single (bundled) dataset on the user's interested period, including moving averaged datasets and making plots.

    usage: **CRMS2Plot.py [-h] [--sdate SDATE] [--edate EDATE] [--staionfile STAIONFILE] [--data_type DATA_TYPE] [--save] [--plotdata PLOTDATA] [--specify_MA SPECIFY_MA]**

    **options:**\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    -h, --help\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;               Show this help message and exit\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    --sdate SDATE\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;            State date for the data analysis (format: YYYY-MM-DD) [Default: "2008-01-01"]\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    --edate EDATE\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;           End date for the data analysis (format: YYYY-MM-DD)[Default: "2024-12-31"]\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    --staionfile STAIONFILE\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Path to station list file <station_list.txt> (format: CRMSxxxx)[Default: None]\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    --data_type DATA_TYPE\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    Data type: hourly(H), daily(D), monthly(M), and yearly(Y)[Default: M]\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    --save\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                  Save as a single (bundled) dataset and MA_datasets. This is time-consuming when the user uses high spatial datasets. [Default: True]\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    --plotdata PLOTDATA\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   Plot original data (org) or moving average data (MA)[Default: MA]\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    --specify_MA SPECIFY_MA\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [Optional] The user can specify a central moving average window size in days. [Default: yearly averaged]\

### Plot examples

<img src="https://github.com/jinikeda/CRMS2Map/blob/main/Image/Water_level_median.png" alt="Long-term water level" width="400">

<p style="text-align: left;"><strong>Figure.1</strong> Median water levels across coastal Louisiana for long-term water level change study using a 12-month moving average window. The inter-quantile range between Q1 and Q3 is shaded grey. Command: CRMS2Plot</p>

<p align="left">
  <img src="https://github.com/jinikeda/CRMS2Map/blob/main/Image/Water_depth_multi_stations.png" alt="Ida_inundation depth"  width="400">
  <img src="https://github.com/jinikeda/CRMS2Map/blob/main/Image/Salinity_multi_stations.png" alt="Ida salinity"  width="400">
</p>
<p style="text-align: left;"><strong>Figure.2</strong> Inundation depth (left) and salinity (right) at multiple CRMS stations during Hurricane Ida in 2021. Command: CRMS2Plot --sdate 2021-08-28 --edate 2021-09-03 --plotdata org --save False --data_type H --staionfile station_list.txt</p>

### Running the Script as a module 
Not recommended because this may result in unpredictable behavior\
***python -m src.Python_Filename_withoutSuffix*** \
e.g., *python -m src.CRMS_Continuous_Hydrographic2subsets* 

### Release history
CRMS2Map v1.0: First release on Aug/08/2024 (Only release CRMS2Plot. Interpolation and mapping tools will be released after paper publication)
