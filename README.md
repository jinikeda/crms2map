# CRMS2Map
Data analytical and mapping tools for the Louisiana Coastal Reference Monitoring System (CRMS) Hydrographic Data from https://cims.coastal.la.gov/monitoring-data/

# Step 1. Install virtual conda env and activation
conda env create -f env.yml

# Step 2. Activate virtual env
conda activate CRMS2Map

## Crate a package
pip install -e . #-e: edding mode

### Contents of the package
#### Step 1. Auto retrieve the original datasets
    Hydrographic Hourly Data: Hourly hydrographic data (over 60GB) -> CRMS_Continuous_Hydrographic2subsets.py (*need largememory for processing)
    Hydrographic Monthly Data: Monthly hydrographic data (Manually corrected such as pore water salinity) -> CRMS_Discrete_Hydrographic2subsets.py
#### Step 2. Resample the retived data
    From the step 1, resample hourly, daily, monthly and yearly -> CRMS2Resample.py
#### Step 2_branch. Plot time series data    

### Running a package
CRMS2Map_continuous for step. 1: CRMS_Continuous_Hydrographic2subsets.py
CRMS2Map_discrete for step.1: CRMS_Discrete_Hydrographic2subsets.py
CRMS2Map_resample for step.2: CRMS2Resample.py

### Running the Script
python -m src.CRMS_Continuous_Hydrographic2subsets
