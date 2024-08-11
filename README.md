# CRMS2Map
Data analytical and mapping tools for the Louisiana Coastal Reference Monitoring System (CRMS) Hydrographic Data from https://cims.coastal.la.gov/monitoring-data/

# §1. Install virtual conda env and activation
conda env create -f env.yml #black is an uncompromising Python code formatter, flack8 is a linter to check code, and pytest and mock are code testing tools. These packages are not mandatory for running CRMS2Map  

# §2. Activate virtual env
conda activate CRMS2Map

# §3 Crate a package
pip install -e . #-e: editable mode (Prefer for package developers)

### Contents of the package
#### Step 1. Auto retrieve the original datasets
    Hydrographic Hourly Data: Hourly hydrographic data (over 60GB) -> CRMS_Continuous_Hydrographic2subsets.py (*need largememory for processing)
    Hydrographic Monthly Data: Monthly hydrographic data (Manually corrected such as pore water salinity) -> CRMS_Discrete_Hydrographic2subsets.py
#### Step 2. Resample the retived data
    From the step 1, resample hourly, daily, monthly and yearly -> CRMS2Resample.py
#### Step 3. Plot time series data (optional)    

### Running a package
**CRMS2Map_continuous** for step. 1: CRMS_Continuous_Hydrographic2subsets.py
**CRMS2Map_discrete** for step.1: Create a subset data for salinity, water temperature, water level, and water level to marsh (hydroperiod and inundation depth) using CRMS_Discrete_Hydrographic2subsets.py
**CRMS2Map_resample** for step.2: Create hourly, daily, monthly, and yearly averaged datasets using CRMS2Resample.py
CRMS2Plot: Data analysis and plotting tool after step.2

    usage: **CRMS2Plot.py [-h] [--sdate SDATE] [--edate EDATE] [--staionfile STAIONFILE] [--data_type DATA_TYPE] [--save] [--plotdata PLOTDATA] [--specify_MA SPECIFY_MA]**

    options:
    -h, --help            show this help message and exit
    --sdate SDATE         State date for the data analysis (format: YYYY-MM-DD)
    --edate EDATE         End date for the data analysis (format: YYYY-MM-DD)
    --staionfile STAIONFILE
                            Path to station list file <station_list.txt> (format: CRMSxxxx)
    --data_type DATA_TYPE
                            Data type: houly(H), daily(D), monthly(M), and yearly(Y)
    --save                Save as a single (bundled) dataset and MA_datasets. This is time-consuming when the user uses high spatial datasets.
    --plotdata PLOTDATA   Plot original data (org) or moving average data (MA)
    --specify_MA SPECIFY_MA
                            [Optional] The user can specify a central moving average window size in days. [Default] = yearly averaged

### Running the Script
~~python -m src.CRMS_Continuous_Hydrographic2subsets~~


### Release history
CRMS2Map v1.0: First release on Aug/08/2024 (Only release CRMS2Plot. Interpolation and mapping tools will be released after paper publication)
