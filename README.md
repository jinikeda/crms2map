# CRMS2Map
Data analytical and mapping tools for the Louisiana Coastal Reference Monitoring System (CRMS) Hydrographic Data from https://cims.coastal.la.gov/monitoring-data/
### CRMS Water Level Data is used Geoid12B after Oct 2, 2023 (no more Geoid12A data are available)

## §1. Install virtual conda env and activation
Type: ***conda env create -f env.yml*** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; black is an uncompromising Python code formatter, flake8 is a linter to check code, and pytest and mock are code testing tools. These packages are not mandatory for running CRMS2Map.  

## §2. Activate virtual env
Type: ***conda activate CRMS_env***

## §3 Create a package: CRMS2Map
Type: ***pip install -e .*** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -e: editable mode (Preferred for package developers)

## Contents of the package
### Step 1. Auto-retrieve the original datasets

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Hydrographic Hourly Data: Hourly hydrographic data (over 60GB). See src/CRMS_Continuous_Hydrographic2subsets.py \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <mark>**[Caution] Need large memory (RAM >= 32 GB) for processing**</mark> \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Hydrographic Monthly Data: Monthly hydrographic data (Manually corrected such as pore water salinity). See src/CRMS_Discrete_Hydrographic2subsets.py

### Step 2. Resample the retrieved data

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; From step 1, resample hourly, daily, monthly, and yearly averaged datasets in the Input folder. The processed and output data are also available in each folder. See src/CRMS2Resample.py

### Step 3. Plot time series data (optional)    

## Running submodule in the CRMS2Map package
* ***CRMS2Map --help*** Show available commands:
  
* ***CRMS2Map continuous*** for step. 1a: CRMS_Continuous_Hydrographic2subsets.py
* ***CRMS2Map discrete*** for step.1b: Create a subset data for salinity, water temperature, water level, and water level to marsh (hydroperiod and inundation depth) using CRMS_Discrete_Hydrographic2subsets.py
* ***CRMS2Map resample*** for step.2: Create hourly, daily, monthly, and yearly averaged datasets using CRMS2Resample.py
* ***CRMS2Map plot***: Data analysis and plotting tool after step.2\
  This tool creates a single (bundled) dataset on the user's interested period, including moving averaged datasets and making plots. 

    usage: [--sdate] [--edate] [--staionfile] [--data_typ] [--save] [--plotdata] [--specify_ma]** \
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <mark>**[Caution] no uppercases are availeble in click**</mark> 

    **Options:** 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    --sdate \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;            State date for the data analysis (format: YYYY-MM-DD) [Default: "2008-01-01"]\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    --edate \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;           End date for the data analysis (format: YYYY-MM-DD)[Default: "2024-12-31"]\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    --staionfile \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Path to station list file <station_list.txt> (format: CRMSxxxx)[Default: None]\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    --data_type \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    Data type: hourly(H), daily(D), monthly(M), and yearly(Y)[Default: M]\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    --save \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                  Save as a single (bundled) dataset and MA_datasets. This is time-consuming when the user uses high spatial datasets. [Default: True] \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    --plotdata \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   Plot original data (org) or moving average data (MA)[Default: MA]\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    --specify_ma \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [Optional] The user can specify a central moving average window size in days. [Default: yearly averaged]

### Plot examples

<img src="https://github.com/jinikeda/CRMS2Map/blob/main/Image/Water_level_median.png" alt="Long-term water level" width="400">

<p style="text-align: left;"><strong>Figure.1</strong> Median water levels across coastal Louisiana for long-term water level change study using a 12-month moving average window. The inter-quantile range between Q1 and Q3 is shaded grey. Command: CRMS2Map plot</p>

<p align="left">
  <img src="https://github.com/jinikeda/CRMS2Map/blob/main/Image/Water_depth_multi_stations.png" alt="Ida_inundation depth"  width="400">
  <img src="https://github.com/jinikeda/CRMS2Map/blob/main/Image/Salinity_multi_stations.png" alt="Ida salinity"  width="400">
</p>
<p style="text-align: left;"><strong>Figure.2</strong> Inundation depth (left) and salinity (right) at multiple CRMS stations during Hurricane Ida in 2021. Command: CRMS2Map plot --sdate 2021-08-28 --edate 2021-09-03 --plotdata org --save False --data_type H --stationfile station_list.txt</p>

### Supplement 
This folder provides data processing codes and additional datasets for wetland hydrological analysis (for the details, read readme.txt in the Supplement folder). \
The **documentation.pdf** contains further dataset background and explanation of CRMS2Map modules and supplemental analysis.

### Release history
CRMS2Map v1.0: First release on Aug/08/2024 (Only release CRMS2Plot. **Interpolation and mapping tools** will be released after paper publication) \
CRMS2Map v1.1: Modified Geoid system and added supplemental analysis codes on Dec/06/2024
