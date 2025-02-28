**CRMS2Map documentation**

**Jin Ikeda**

**Last Modify 01/09/25**

CRMS2Map: Data analytical and mapping tools for the Louisiana Coastal
Reference Monitoring System (CRMS)

Repository: <https://github.com/jinikeda/crms2map>

**Datasets**
------------

Available data list <https://cims.coastal.la.gov/monitoring-data/>

Bulk downloads: <https://cims.coastal.la.gov/FullTableExports.aspx>

CRMS data list

-   Hydrographic Data (Continuous Hydrographic (Hourly), Discrete
    Hydrographic (Monthly)) using bulk download links

-   Coastal Basin (GIS data -\> Reference layers -\> **Basins**)

> Input/Basin\_NAD83.shp (ten coastal domains defined by CPRA)
>
> Input/Basin\_NAD83\_Dissolve.shp (merged single domain)

-   Datum conversion for surface water elevation (**Geoid12B** after Oct
    2, 2023. No more Geoid12A data are available)

> <https://www.lacoast.gov/chart/Charting.aspx?laf=crms&tab=2>
>
> ![A screenshot of a computer Description automatically
> generated](media/image1.png){width="3.57in" height="3.0in"}

**Environment Setting**
-----------------------

Follow sections 1 to 3 in the Readme file:
<https://github.com/jinikeda/crms2map>

**CRMS2 Map package: Features and Workflow**
--------------------------------------------

### **1. Auto-Retrieve Hydrographic Data**

-   **CRMS2Map continuous**:

    -   Retrieves and subsets continuous hydrographic data (hourly).

    -   Estimated Time: 10 -- 20 minutes.

-   **CRMS2Map discrete**:

    -   Retrieves and subsets discrete hydrographic data (monthly).

    -   Estimated Time: \~1 minute.

### **2. Data Processing of Hydrographic Data**

-   **CRMS2Map resample**:

    -   Generates averaged datasets (hourly, daily, monthly, yearly)
        from continuous and discrete hydrographic data.

    -   Processed datasets are saved in the "Input" folder.

    -   Processed and output data are organized in their respective
        folders for easy access.

    -   Estimated Time: \~3 minutes.

### **3. Data Visualization of Hydrographic Data**

-   **CRMS2Map plot**:

    -   Creates time-series plots for the user's specified period.

    -   Includes (moving-averaged) datasets for:

        -   Salinity \[ppt\]

        -   Water level \[m, NAVD88\]

        -   Percent time inundation/Hydro period \[-\]

        -   Inundation depth \[m\]

    -   **Station Specification**:

        -   When the user wants to specify the station(s), of interest,
            update the station\_list.txt file located in the parent
            folder.

    -   **Estimated Time**: \~2 minutes.

> option
>
> \--sdate State date for the data analysis (format: YYYY-MM-DD)
> \[Default: \"2008-01-01\"\]\
> \--edate End date for the data analysis (format: YYYY-MM-DD)\[Default:
> \"2024-12-31\"\]\
> \--staionfile Path to station list file \<station\_list.txt\> (format:
> CRMSxxxx)\[Default: None\]\
> \--data\_type Data type: hourly(H), daily(D), monthly(M), and
> yearly(Y)\[Default: M\]\
> \--save Save as a single (bundled) dataset and MA\_datasets. This is
> time-consuming when the user uses high spatial datasets. \[Default:
> True\]\
> \--plotdata Plot original data (org) or moving average data
> (MA)\[Default: MA\]\
> \--specify\_ma \[Optional\] The user can specify a central moving
> average window size in days. \[Default: yearly averaged\]

**CRMS2Map Pytest**
-------------------

-   ### tests/test\_CRMS\_general\_functions.py 

```{=html}
<!-- -->
```
-   Test individual functions used in each submodule (Github Action
    automatically tests the CI/CD pipeline when changing the code).

**\
**

**Supplement Analysis** (folder: CRMS2Map/Supplement)
-----------------------------------------------------

**Datasets**
------------

-   **CRMS station coordinates**

> CRMS\_Long\_Lat.csv
>
> <https://www.lacoast.gov/crms_viewer/Map/CRMSViewer> -\> Download
> Lat./Long.
>
> ![](media/image2.jpeg){width="5.3in" height="3.0in"}

-   **Marsh Vegetation**

> (<https://cims.coastal.la.gov/FullTableExports.aspx> -\> Full Table
> Exports - CRMS Data Only -\> **Marsh Vegetation**)
>
> Processed CRMS\_Long\_Lat.csv and CRMS\_Marsh\_Vegetation.csv to make
> dominant marsh vegetation types and the station's coastal domains.

-   Location: Data/CRMS\_station\_Basin\_Community.shp

```{=html}
<!-- -->
```
-   **Station list for each variable** (Surface\_salinity, water
    elevation, temperature, etc)

```{=html}
<!-- -->
```
-   Data/CRMS\_stations\_Surface\_salinity.shp

-   Data/CRMS\_stations\_Water\_Elevation\_to\_Datum.shp

-   Data/CRMS\_stations\_Water\_Elevation\_to\_Marsh.shp

-   Data/CRMS\_stations\_Water\_Temp.shp

```{=html}
<!-- -->
```
-   **Median monthly climate drivers** (sea surface temperature, river
    flow, precipitation, winds) in Coastal Louisiana (Processed data)

```{=html}
<!-- -->
```
-   Locations: Data/MonthlySST.xlsx

> ![](media/image3.emf){width="5.60592738407699in"
> height="0.7840004374453193in"}
>
> Subdomain/community analysis:

-   Data/AR\_daily\_discharge\_since\_1970.csv (daily Atchafalaya River
    discharge)

-   Data/CS\_discharge\_since\_2008.csv (daily Calcasieu River
    discharge)

-   Data/Basin\_total\_prcp Monthly.xlsx (total precipitation in each
    basin)

For the detailed datasets list, please refer to Table2 on
*"Tempo-spatial variations in water level and salinity in Louisiana
coastal wetlands over 15 years"*

### **1. Function of Monthly\_analysis\_practice.py**

-   Data Reading and Preparation

    -   Read monthly continuous and discrete hydrographic datasets.

-   Data Analysis

    -   Generate 12-month moving average datasets.

    -   Examine short- (15 years) and long-term (over 40 years) trends
        for climate driver and CRMS data.

-   Data Grouping

    -   Grouped by subdomain and vegetation datasets.

-   Data Visualization

    -   Generates visualizations for subdomain and vegetation-specific
        datasets.

-   Statistical Analysis

    -   Analyzes correlations between subdomain/vegetation datasets and
        climate drivers.

> **Outputs**:

-   Plots for subdomains and vegetation datasets (e.g., Photo folder).

-   Display/output statistical results (e.g., Sub\_basin/Sub\_marsh
    > folders).

### **2. Function of Bootstrap\_Regression\_analysis.py**

-   Multiple regression models

    -   Automated bootstrap regression analysis using ordinal linear and
        random forest models.

-   Statistical Analysis

    -   Evaluate the performance of models.

> **Outputs**:

-   Statistical results (e.g. bootstrap\_Output folder).

### **3. Function of Regression\_analysis\_plot.py**

-   Data Visualization

    -   Generates a time series of visualizations for each subdomain.

-   Statistical Analysis

    -   Generates a summary table of model performance for each
        subdomain.

> **Outputs**:

-   Plot a time series of model predictions and comparisons (Photo
    > folder).

-   Generate model performance and statistical results
    > (bootstrap\_Output folder).
