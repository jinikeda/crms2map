#!/usr/bin/env python
# coding: utf-8
# CRMS2Plot for time series analysis
# This file is developed for monthly data analysis. However, it includes hourly and daily analysis capabilities.
# Developed by the Center for Computation & Technology at Louisiana State University (LSU).
# Developer: Jin Ikeda
# Last modified Aug 10, 2024
import os

# Import necessary modules
from datetime import datetime, timedelta
import argparse

# stats

# plot
import matplotlib.dates as mp_dates
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import itertools

from CRMS_general_functions import *

start_time = time.time()

global file_suffix, file_suffix2

file_suffix = ".csv"
file_suffix2 = ".xlsx"

# Declare global variables
start_date = None
end_date = None

# Make Output folder
Workspace = "C:/Users/jikeda/Desktop/CRMS2Map/CRMS_devtest"
Inputspace = os.path.join(Workspace, "Input")  # Make Input folder

########################################################################################################################
# Set the list of CRMS stations to plot

### Manual input ###
station_list = ["CRMS0002", "CRMS0003"]

### Use txt file ###
### user turn on the following code to use the txt file
# Inputspace = os.path.join(Workspace, "Input")  # Input folder
# df = pd.read_csv(os.path.join(Inputspace,'CRMS_Water_Temp_2006_2024_Mdata.csv'), encoding='utf-8')
# station_list= df.columns[1:-1].tolist() # exclude 'Date' and 'num_station' columns
# np.savetxt(os.path.join(Inputspace,'station_list.txt'), station_list, fmt='%s')

# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))

# Change the current working directory
os.chdir(Workspace)

Photospace = os.path.join(Workspace, "Photo")
Outputspace = os.path.join(Workspace, "Output")
# Output_space=os.path.join(Workspace,'bootstrap_Output')

try:
    os.makedirs(Photospace, exist_ok=True)
    os.makedirs(Outputspace, exist_ok=True)
except:
    pass


### Functions ####################################################################################################
# Make a nested datasets for continuous data
def create_nested_datasets(
    file_name, file_name_o, file_suffix, MA_window, threshold1, Discrete=False
):
    global start_date, end_date  # Refer to the global variables
    datasets = {}  # monthly average dataset
    MA_datasets = {}  # moving average dictionaly
    for file_n, name_o in zip(file_name, file_name_o):
        file = file_n + file_suffix
        print(file)
        try:
            datasets[name_o] = pd.read_csv(file, encoding="utf-8")
        except UnicodeDecodeError:
            print("Encoding error")
            continue

        print(datasets[name_o].head(5))

        # Check data size and type
        print(datasets[name_o].shape, datasets[name_o].dtypes)
        datasets[name_o] = pd.DataFrame(datasets[name_o])

        # Delete columns where 'num_station' is lower than threshold1
        row_index = (
            datasets[name_o]
            .loc[datasets[name_o]["num_station"] < threshold1]
            .index.tolist()
        )
        datasets[name_o] = datasets[name_o].drop(row_index)

        # Move datetime into index
        datasets[name_o].index = pd.to_datetime(datasets[name_o].Date)

        # Drop the columns that were used to create the index
        datasets[name_o].drop(["Date"], axis=1, inplace=True)
        datasets[name_o].drop(["num_station"], axis=1, inplace=True)
        datasets[name_o] = datasets[name_o].iloc[2:, :]

        # Calculate moving average
        if Discrete == False:
            MA_datasets[name_o] = (
                datasets[name_o].rolling(window=MA_window, center=True).mean()
            )  # , min_periods=9min_period is a rough criterion
        else:
            MA_datasets[name_o] = (
                datasets[name_o]
                .rolling(window=MA_window, center=True, min_periods=int(MA_window / 2))
                .mean()
            )  # , min_periods=9min_period is a rough criterion

        # Filtering the data
        datasets[name_o] = datasets[name_o].query(
            "index >= @start_date and index <= @end_date"
        )
        MA_datasets[name_o] = MA_datasets[name_o].query(
            "index >= @start_date and index <= @end_date"
        )

    return datasets, MA_datasets


def create_dataframe(file_name, date_column="Date"):

    # Use the appropriate pandas function to read the file
    if ".xlsx" in file_name:
        try:
            df = pd.read_excel(file_name)
        except UnicodeDecodeError:
            print("encoding error")  # CRMS = pd.read_csv(file, encoding='utf-8')
    elif ".csv" in file_name:
        try:
            df = pd.read_csv(file_name)
        except UnicodeDecodeError:
            print("encoding error")  # CRMS = pd.read_csv(file, encoding='utf-8')
    else:
        raise ValueError(
            "Unsupported file type. The file must be a .xlsx or .csv file."
        )

    # Convert the DataFrame to the correct format
    df = pd.DataFrame(df)

    print(df.head(5))
    print(df.shape, df.dtypes)  # Check data size and type

    # Set the index to the 'Date' column and drop the original 'Date' column
    df.index = pd.to_datetime(df[date_column])
    df.drop([date_column], axis=1, inplace=True)

    # Future revision may consider filtering of datasets

    return df


def sub_dataframe_gen(sub_name, file_name):
    sub_datasets = {}  # monthly average dataset

    for file, name_o in zip(sub_name, file_name):
        print(file)
        sub_datasets[name_o] = create_dataframe(file, "Date")

    return sub_datasets


# Plot CRMS data
def plot_CRMS(
    datasets_continous,
    datasets_discrete,
    file_name_o,
    plot_period,
    plot_space,
    plot_range=None,
    station=None,
):
    """
    General function to plot water level, hydroperiod, inundation depth, salinity, and other environmental data.

    Parameters:
    - datasets: dict of datasets
    - datasets_continous: dict containing main datasets (water level, salinity, etc.)
    - datasets_discrete: dict containing discrete datasets (for pore water, etc.)
    - file_name_o: str, the type of plot ("WL", "Salinity", "W_HP", etc.)
    - plot_period: list or tuple, the period over which to plot the data
    - plot_range: list or tuple, the y-axis range for the plot
    - plot_space: float, the spacing of y-axis ticks. If the value is less than plot_range/4, automatically modify the value
    - station: str or list, optional, the CRMS station(s) to plot data for. If None, plots the median across all stations.
    """

    ####################################################################################################################
    # Automatically define the plot_range based on data (if the range does not provided by the user)
    ####################################################################################################################
    if plot_range is None:
        if station:
            if isinstance(station, list):
                min_value = round(
                    datasets_continous[file_name_o][station].min().min() - 0.05, 1
                )
                max_value = round(
                    datasets_continous[file_name_o][station].max().max() + 0.05, 1
                )
            else:
                min_value = round(
                    datasets_continous[file_name_o][station].min() - 0.05, 1
                )
                max_value = round(
                    datasets_continous[file_name_o][station].max() + 0.05, 1
                )
        else:
            # Calculate the 10th percentile at each time step
            quantiles_low = datasets_continous[file_name_o].quantile(q=0.25, axis=1)
            quantiles_high = datasets_continous[file_name_o].quantile(q=0.75, axis=1)

            # Take the minimum and maximum of these quantiles across all time steps
            min_value = round(quantiles_low.min() - 0.05, 1)
            max_value = round(quantiles_high.max() + 0.05, 1)

        if np.max(max_value) > 1:
            plot_range = [
                np.floor(min_value),
                np.ceil(max_value),
            ]  # Provide plot_range as a list
        else:
            plot_range = [min_value, max_value]  # Provide plot_range as a list
        print("Min_range:", plot_range[0], "Max_range:", plot_range[1])

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlabel("Year")

    ####################################################################################################################
    # Automatically set up the x-axis for date formatting
    ####################################################################################################################
    # Convert plot_period from strings to datetime objects
    start_date = datetime.strptime(plot_period[0], "%Y-%m-%d")
    end_date = datetime.strptime(plot_period[1], "%Y-%m-%d")

    # Calculate data length
    if Data_type == "H":
        data_length = (end_date - start_date).total_seconds() / (
            3600 * 24
        )  # Data length in days
        if data_length <= 1:
            print(f"Data Length: {data_length*24} hours")
        else:
            print(f"Data Length: {data_length} days")
    elif Data_type == "D":
        data_length = (end_date - start_date).days  # Data length in days
        print(f"Data Length: {data_length} days")
    else:
        data_length = (end_date - start_date).days / 365.25  # Data length in years
        print(f"Data Length: {data_length} years" )

    # Set x-axis date formatter based on the data length
    assert (
        data_length > 0
    ), "Invalid data period. The end date must be after the start date."



    # Set x-axis date formatter and locator based on the data length and type
    if Data_type == "H":
        if data_length <= 0.25:
            ax.xaxis.set_major_formatter(mp_dates.DateFormatter("%Y-%m-%d-%H"))
            ax.xaxis.set_major_locator(
                mp_dates.HourLocator(interval=2)
            )  # Every 2 hours
        elif data_length <= 1:
            ax.xaxis.set_major_formatter(mp_dates.DateFormatter("%Y-%m-%d-%H"))
            ax.xaxis.set_major_locator(
                mp_dates.HourLocator(interval=6)
            )  # Every 6 hours
        elif data_length <= 3:
            ax.xaxis.set_major_formatter(mp_dates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mp_dates.DayLocator(interval=1))  # Every day
        elif data_length <= 7:
            ax.xaxis.set_major_formatter(mp_dates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mp_dates.DayLocator(interval=2))  # Every 2 days
        elif data_length <= 31:
            ax.xaxis.set_major_formatter(mp_dates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mp_dates.DayLocator(interval=7))  # Every 7 days
        elif data_length <= 120:
            ax.xaxis.set_major_formatter(mp_dates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mp_dates.MonthLocator(interval=1))  # Every month
        elif data_length <= 367:
            ax.xaxis.set_major_formatter(mp_dates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(
                mp_dates.MonthLocator(interval=3)
            )  # Every 3 months
        else:
            ax.xaxis.set_major_formatter(mp_dates.DateFormatter("%Y"))
            ax.xaxis.set_major_locator(
                mp_dates.YearLocator(base=max(1, round(data_length / (365.25 * 4))))
            )
    elif Data_type == "D":
        if data_length <= 7:
            ax.xaxis.set_major_formatter(mp_dates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mp_dates.DayLocator(interval=2))  # Every 2 days
        elif data_length <= 31:
            ax.xaxis.set_major_formatter(mp_dates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mp_dates.DayLocator(interval=7))  # Every 7 days
        elif data_length <= 120:
            ax.xaxis.set_major_formatter(mp_dates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mp_dates.MonthLocator(interval=1))  # Every month
        elif data_length <= 367:
            print("check the selection month")
            ax.xaxis.set_major_formatter(mp_dates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(
                mp_dates.MonthLocator(interval=3)
            )  # Every 3 months
        else:
            print("check the selection")
            ax.xaxis.set_major_formatter(mp_dates.DateFormatter("%Y"))
            ax.xaxis.set_major_locator(
                mp_dates.YearLocator(base=max(1, round(data_length / (365.25 * 4))))
            )
    else:  # Monthly or yearly data
        if data_length <= 1.5:
            ax.xaxis.set_major_formatter(mp_dates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(
                mp_dates.MonthLocator(interval=3)
            )  # Every 3 months
        elif data_length <= 3:
            ax.xaxis.set_major_formatter(mp_dates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(
                mp_dates.MonthLocator(interval=6)
            )  # Every 6 months
        else:
            ax.xaxis.set_major_formatter(mp_dates.DateFormatter("%Y"))
            ax.xaxis.set_major_locator(
                mp_dates.YearLocator(base=max(1, round(data_length / 4)))
            )

    if Data_type == "H" or Data_type == "D":
        ax.set_xlabel("Date")
        # Rotate the x-axis labels for better readability
        if data_length <= 5:
            plt.xticks(rotation=45, ha="right")
    else:
        ax.set_ylabel("Year")

    ####################################################################################################################
    # handle specified stations
    ####################################################################################################################

    if station:
        if isinstance(station, list):
            title_suffix = f" at {len(station)} stations"
            if len(station) >= 5:
                title_out = "multi_stations"
            else:
                title_out = "_".join([s.replace("CRMS", "") for s in station])
        else:
            title_suffix = f" at {station.replace('CRMS', '')}"
            title_out = station.replace("-", " ")
    else:
        title_suffix = " (Median Across Stations)"
        title_out = "median"

    if file_name_o == "WL":
        ax.set_ylabel("Water level [NAVD88,m]")
        output = os.path.join(Photospace, f"Water_level_{title_out}.png")
        if station:
            if isinstance(station, list):
                for st in station:
                    ax.plot(
                        datasets_continous["WL"].index,
                        datasets_continous["WL"][st],
                        label=f"{st}",
                        linewidth=1,
                    )
            else:
                ax.plot(
                    datasets_continous["WL"].index,
                    datasets_continous["WL"][station],
                    "k--",
                    linewidth=1,
                )
        else:
            ax.plot(
                datasets_continous["WL"].index,
                datasets_continous["WL"].median(axis=1, skipna=True),
                "k--",
                linewidth=1,
            )
            plt.fill_between(
                datasets_continous["WL"].index,
                datasets_continous["WL"].quantile(q=0.25, axis=1),
                datasets_continous["WL"].quantile(q=0.75, axis=1),
                alpha=0.9,
                linewidth=0,
                color="grey",
            )

    elif file_name_o == "Temp":
        ax.set_ylabel("Temperature [°C]")
        output = os.path.join(Photospace, f"Temperature_{title_out}.png")
        if station:
            if isinstance(station, list):
                for st in station:
                    ax.plot(
                        datasets_continous["Temp"].index,
                        datasets_continous["Temp"][st],
                        label=f"{st}",
                        linewidth=1,
                    )
            else:
                ax.plot(
                    datasets_continous["Temp"].index,
                    datasets_continous["Temp"][station],
                    "k--",
                    linewidth=1,
                )
        else:
            ax.plot(
                datasets_continous["Temp"].index,
                datasets_continous["Temp"].median(axis=1, skipna=True),
                "k--",
                linewidth=1,
            )
            plt.fill_between(
                datasets_continous["Temp"].index,
                datasets_continous["Temp"].quantile(q=0.25, axis=1),
                datasets_continous["Temp"].quantile(q=0.75, axis=1),
                alpha=0.9,
                linewidth=0,
                color="grey",
            )

    elif file_name_o == "W_HP":
        ax.set_ylabel("Hydroperiod")
        output = os.path.join(Photospace, f"Hydro_period_{title_out}.png")
        if station:
            if isinstance(station, list):
                for st in station:
                    ax.plot(
                        datasets_continous["W_HP"].index,
                        datasets_continous["W_HP"][st],
                        label=f"{st}",
                        linewidth=1,
                    )
            else:
                ax.plot(
                    datasets_continous["W_HP"].index,
                    datasets_continous["W_HP"][station],
                    "k--",
                    linewidth=1,
                )
        else:
            ax.plot(
                datasets_continous["W_HP"].index,
                datasets_continous["W_HP"].median(axis=1, skipna=True),
                "k--",
                linewidth=1,
            )
            plt.fill_between(
                datasets_continous["W_HP"].index,
                datasets_continous["W_HP"].quantile(q=0.25, axis=1),
                datasets_continous["W_HP"].quantile(q=0.75, axis=1),
                alpha=0.9,
                linewidth=0,
                color="grey",
            )

    elif file_name_o == "W_depth":
        ax.set_ylabel("Inundation depth [m]")
        output = os.path.join(Photospace, f"Water_depth_{title_out}.png")
        if station:
            if isinstance(station, list):
                for st in station:
                    ax.plot(
                        datasets_continous["W_depth"].index,
                        datasets_continous["W_depth"][st],
                        label=f"{st}",
                        linewidth=1,
                    )
            else:
                ax.plot(
                    datasets_continous["W_depth"].index,
                    datasets_continous["W_depth"][station],
                    "k--",
                    linewidth=1,
                )
        else:
            ax.plot(
                datasets_continous["W_depth"].index,
                datasets_continous["W_depth"].median(axis=1, skipna=True),
                "k--",
                linewidth=1,
            )
            plt.fill_between(
                datasets_continous["W_depth"].index,
                datasets_continous["W_depth"].quantile(q=0.25, axis=1),
                datasets_continous["W_depth"].quantile(q=0.75, axis=1),
                alpha=0.9,
                linewidth=0,
                color="grey",
            )

    elif file_name_o == "Salinity":
        ax.set_ylabel("Salinity [ppt]")
        output = os.path.join(Photospace, f"Salinity_{title_out}.png")
        if station:
            if isinstance(station, list):
                for st in station:
                    ax.plot(
                        datasets_continous["Salinity"].index,
                        datasets_continous["Salinity"][st],
                        label=f"{st}",
                        linewidth=1,
                    )

                    # User can change the display option

                    # if Data_type == "M" or Data_type == "Y":
                    #     ax.plot(
                    #         datasets_discrete["Pore_10"].index,
                    #         datasets_discrete["Pore_10"][st],
                    #         "g--",
                    #         linewidth=1,
                    #     )
                    #     ax.plot(
                    #         datasets_discrete["Pore_30"].index,
                    #         datasets_discrete["Pore_30"][st],
                    #         "r--",
                    #         linewidth=1,
                    #     )
            else:
                ax.plot(
                    datasets_continous["Salinity"].index,
                    datasets_continous["Salinity"][station],
                    "k--",
                    linewidth=1,
                )
                if Data_type == "M" or Data_type == "Y":
                    ax.plot(
                        datasets_discrete["Pore_10"].index,
                        datasets_discrete["Pore_10"][station],
                        "g--",
                        linewidth=1,
                    )
                    ax.plot(
                        datasets_discrete["Pore_30"].index,
                        datasets_discrete["Pore_30"][station],
                        "r--",
                        linewidth=1,
                    )
        else:
            ax.plot(
                datasets_continous["Salinity"].index,
                datasets_continous["Salinity"].median(axis=1, skipna=True),
                "k--",
                linewidth=1,
            )
            if Data_type == "M" or Data_type == "Y":
                ax.plot(
                    datasets_discrete["Pore_10"].index,
                    datasets_discrete["Pore_10"].median(axis=1, skipna=True),
                    "g--",
                    linewidth=1,
                )
                ax.plot(
                    datasets_discrete["Pore_30"].index,
                    datasets_discrete["Pore_30"].median(axis=1, skipna=True),
                    "r--",
                    linewidth=1,
                )

                plt.fill_between(
                    datasets_discrete["Pore_30"].index,
                    datasets_discrete["Pore_30"].quantile(q=0.25, axis=1),
                    datasets_discrete["Pore_30"].quantile(q=0.75, axis=1),
                    alpha=0.5,
                    linewidth=0,
                    color="r",
                )
                plt.fill_between(
                    datasets_discrete["Pore_10"].index,
                    datasets_discrete["Pore_10"].quantile(q=0.25, axis=1),
                    datasets_discrete["Pore_10"].quantile(q=0.75, axis=1),
                    alpha=0.5,
                    linewidth=0,
                    color="g",
                )
            plt.fill_between(
                datasets_continous["Salinity"].index,
                datasets_continous["Salinity"].quantile(q=0.25, axis=1),
                datasets_continous["Salinity"].quantile(q=0.75, axis=1),
                alpha=0.9,
                linewidth=0,
                color="grey",
            )

        if Data_type == "M" or Data_type == "Y":
            ax.legend(["Pore d30", "Pore d10", "Surface"])
        else:
            ax.legend(["Surface"])

    else:
        raise ValueError(
            "Unsupported file name. The file name must be one of 'WL', 'W_HP', 'W_depth', or 'Salinity'."
        )

    if station:
        plt.text(
            0.05,
            0.95,
            f"{file_name_o} {title_suffix}",
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
        )

    plt.xlim(mp_dates.date2num(plot_period))
    plt.ylim(plot_range)
    if (plot_range[1] - plot_range[0]) / 4 >= plot_space:
        plot_space = round((plot_range[1] - plot_range[0]) / 4, 2)
    major_ticks = np.arange(plot_range[0], plot_range[1] + 0.01, plot_space)
    ax.set_yticks(major_ticks)
    plt.grid(color="k", linestyle="--", linewidth=0.1)
    if isinstance(station, list):
        ax.legend()
    plt.savefig(output, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()


def data_analysis():
    ########################################################################
    print("Data_anaysis")
    ######################################################################

    # Main function to perform data analysis and plotting.
    # Allows users to specify state and date as command line arguments.
    global start_date, end_date, station_list, Data_type  # Refer to the global variables

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="CRMS2Plot: Data analysis and plotting tool"
    )
    parser.add_argument(
        "--sdate",
        type=str,
        default="2008-01-01",
        help="State date for the data analysis (format: YYYY-MM-DD)",
    )
    parser.add_argument(
        "--edate",
        type=str,
        default="2024-12-31",
        help="End date for the data analysis (format: YYYY-MM-DD)",
    )

    parser.add_argument(
        "--staionfile",
        type=str,
        default=None,
        help="Path to station list file <station_list.txt> (format: CRMSxxxx)",
    )

    parser.add_argument(
        "--data_type",
        type=str,
        default="M",
        help="Data type: houly(H), daily(D), monthly(M), and yearly(Y)",
    )

    parser.add_argument(
        "--save",
        type=str2bool,
        default=True,
        help="Save as a single (bundled) dataset and MA_datasets. This is time-consuming when the user uses high spatial datasets.",
    )

    parser.add_argument(
        "--plotdata",
        type=str,
        default="MA",
        help="Plot original data (org) or moving average data (MA)",
    )

    parser.add_argument(
        "--specify_MA",
        type=int,
        default=None,
        help="[Optional] The user can specify a central moving average window size in days. [Default] = yearly averaged",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Extract the parsed arguments
    start_date = args.sdate
    end_date = args.edate
    stationFile = args.staionfile
    Data_type = args.data_type
    Save_Flag = args.save
    plot_data = args.plotdata
    Specified_MA = args.specify_MA

    # Here you can use the `state` and `date` variables in your data analysis logic
    print(f"Performing analysis for between {start_date} - {end_date}")
    # The target working directory

    ### Parameters ###########################################################
    threshold1 = 200  # Delete columns where 'num_station' is lower than the threshold for continuous data
    threshold2 = int(
        threshold1 / 2
    )  # Delete columns where 'num_station' is lower than the threshold for discrete data

    end_date_datetime = datetime.strptime(
        end_date, "%Y-%m-%d"
    )  # Convert the end_date string to a datetime object
    end_date_datetime = end_date_datetime + timedelta(
        days=1
    )  # Add one day to the end_date
    end_date_plus_one = end_date_datetime.strftime(
        "%Y-%m-%d"
    )  # Convert the datetime object back to a string

    # Update the plot_period
    plot_period = [start_date, end_date_plus_one]  # 'yyyy-mm-dd'
    print("plot_period", plot_period)

    Sub_basinspace = os.path.join(Workspace, "Sub_basin")  # Make Sub_basin folder
    Sub_marshspace = os.path.join(Workspace, "Sub_marsh")  # Make Sub_marsh folder

    try:
        # os.makedirs(Photpspace, exist_ok=True)
        os.makedirs(Sub_basinspace, exist_ok=True)
        os.makedirs(Sub_marshspace, exist_ok=True)

    except Exception as e:
        print(f"An error occurred while creating directories: {e}")

    #############################################
    # color_palette for plots
    #############################################
    # Create ramdom colors
    # p.random.seed(42)  # Set the seed for reproducibility
    color_palette = []

    color_palette.append([153 / 255, 0, 0])  # Pontchartrain
    color_palette.append([255 / 255, 200 / 255, 0 / 255])  # Breton Sound
    color_palette.append([255 / 255, 153 / 255, 0 / 255])  # Mississippi River Delta
    color_palette.append([204 / 255, 210 / 255, 0 / 255])  # Barataria
    color_palette.append([0 / 255, 128 / 255, 0 / 255])  # Terrebonne
    color_palette.append([0 / 255, 0 / 255, 255 / 255])  # Atchafalaya
    color_palette.append([153 / 255, 153 / 255, 255 / 255])  # Teche/Vermilion
    color_palette.append([204 / 255, 102 / 255, 153 / 255])  # Mermentau
    color_palette.append([255 / 255, 0, 255 / 255])  # Calcasieu/Sabine

    color_palette_vegetation = []
    # print(color_palette)
    color_palette_vegetation.append([230 / 255, 230 / 255, 0])  # Brackish
    color_palette_vegetation.append([115 / 255, 255 / 255, 223 / 255])  # Freshwater
    color_palette_vegetation.append([223 / 255, 115 / 255, 255 / 255])  # Intermediate
    color_palette_vegetation.append([255 / 255, 85 / 255, 0 / 255])  # Saline
    color_palette_vegetation.append([204 / 255, 204 / 255, 204 / 255])  # Swamp

    ### Step 2 ###########################################################
    print("Step 2: Read input data ")
    ######################################################################

    ### 2.1 Read CRMS files ###

    ### Open continuous files
    if Data_type == "H":
        data_suffix = "Hdata"
    elif Data_type == "D":
        data_suffix = "Ddata"
    elif Data_type == "M":
        data_suffix = "Mdata"
    elif Data_type == "Y":
        data_suffix = "Ydata"

    file_name1 = f"CRMS_Water_Temp_2006_2024_{data_suffix}"
    file_name2 = f"CRMS_Surface_salinity_2006_2024_{data_suffix}"
    file_name3 = f"CRMS_Geoid99_to_Geoid12a_offsets_2006_2024_{data_suffix}"
    file_name4 = f"CRMS_Water_Elevation_to_Marsh_2006_2024_wdepth_{data_suffix}"
    file_name5 = f"CRMS_Water_Elevation_to_Marsh_2006_2024_wd_{data_suffix}"

    file_name = [file_name1, file_name2, file_name3, file_name4, file_name5]
    file_name_o = ["Temp", "Salinity", "WL", "W_depth", "W_HP"]

    ### Open discrete files
    if Data_type == "M" or Data_type == "Y":
        file_name9 = f"Pore_salinity_10_{data_suffix}"
        file_name10 = f"Pore_salinity_30_{data_suffix}"
        file_name_discrete = [file_name9, file_name10]
        file_name_o_discrete = ["Pore_10", "Pore_30"]

    #######################################################################################################################
    # Set a moving window range for yearly analysis
    date_style, MA_window, date_removal = get_date_info(file_name1)
    if Specified_MA:
        MA_window = Specified_MA  # days
        print(f"Specified moving average window size: {MA_window} days")
        if Data_type == "Y":
            MA_window = MA_window // 365
            print(f"Moving average window size: {MA_window} years")
        elif Data_type == "M":
            MA_window = MA_window // 30
            print(f"Moving average window size: {MA_window} months")

        # Future consideration for hourly data: Jin Aug/10/2024

    #######################################################################################################################

    ### 2.2 Open climate file

    # Create a nested dataset for continuous data between 2008-2022
    datasets, MA_datasets = create_nested_datasets(
        file_name, file_name_o, file_suffix, MA_window, threshold1
    )

    if Save_Flag:
        print(Save_Flag)
        print("\n\n Save datasets! Please patience! \n\n")
        with pd.ExcelWriter(os.path.join(Outputspace, "MA_timeseries.xlsx")) as writer:
            for variable in file_name_o:
                MA_datasets[variable].to_excel(writer, sheet_name=variable)

        with pd.ExcelWriter(os.path.join(Outputspace, "timeseries.xlsx")) as writer:
            for variable in file_name_o:
                datasets[variable].to_excel(writer, sheet_name=variable)

    print(
        "##########################################################################################################################\n"
    )
    print("W_HP datasets", datasets["W_HP"])
    print(
        "\n##########################################################################################################################\n"
    )

    # Display stats
    print(
        "HP =",
        datasets["W_HP"].mean().mean(),
        ", Depth= ",
        datasets["W_depth"].mean().mean(),
    )

    # Create a nested dataset for discrete data between 2008-2022
    discrete_datasets = {}  # Initialize the discrete datasets for hourly and daily
    MA_datasets_discrete = (
        {}
    )  # Initialize the moving average datasets for discrete data for hourly and daily
    if Data_type == "M" or Data_type == "Y":
        datasets_discrete, MA_datasets_discrete = create_nested_datasets(
            file_name_discrete,
            file_name_o_discrete,
            file_suffix,
            MA_window,
            threshold2,
            Discrete=True,
        )

        if Save_Flag:
            with pd.ExcelWriter(
                os.path.join(Outputspace, "MA_discrete_timeseries.xlsx")
            ) as writer:
                for variable in file_name_o_discrete:
                    MA_datasets_discrete[variable].to_excel(writer, sheet_name=variable)

            with pd.ExcelWriter(
                os.path.join(Outputspace, "discrete_timeseries.xlsx")
            ) as writer:
                for variable in file_name_o_discrete:
                    datasets_discrete[variable].to_excel(writer, sheet_name=variable)

        print(
            "##########################################################################################################################\n"
        )
        print("W_HP datasets", datasets_discrete["Pore_10"].head(10))
        print(
            "\n##########################################################################################################################\n"
        )

    ### Step 3 ###########################################################
    print(
        "##########################################################################################################################\n"
    )
    print("Step 3: Plot input data ")
    print(
        "\n##########################################################################################################################\n"
    )
    ######################################################################

    if plot_data == "MA":
        contious_datasets = MA_datasets.copy()
        if Data_type == "M" or Data_type == "Y":
            discrete_datasets = MA_datasets_discrete.copy()
    else:
        contious_datasets = datasets.copy()
        if Data_type == "M" or Data_type == "Y":
            discrete_datasets = datasets_discrete.copy()

    if stationFile:
        # assert Inputspace, "The input space must be defined to use the station file."
        # stationFile = os.path.join(Inputspace, stationFile)
        path = os.getcwd()
        assert os.path.exists(
            stationFile
        ), f"The station file {stationFile} does not exist in {path}."
        station_list = np.loadtxt(stationFile, dtype=str).tolist()
        print(f"Using station list from {stationFile}", station_list)

    else:
        station_list = None

    plot_CRMS(contious_datasets, discrete_datasets, "Temp", plot_period, 2, plot_range=None, station=station_list)
    plot_CRMS(
        contious_datasets,
        discrete_datasets,
        "WL",
        plot_period,
        0.1,
        plot_range=None,
        station=station_list,
    )
    plot_CRMS(
        contious_datasets,
        discrete_datasets,
        "Salinity",
        plot_period,
        4,
        plot_range=None,
        station=station_list,
    )

    plot_CRMS(contious_datasets, discrete_datasets, "W_HP", plot_period, 0.2,plot_range=None,station=station_list)
    plot_CRMS(contious_datasets, discrete_datasets, "W_depth", plot_period, 0.1,plot_range=None,station=station_list)

    # ### Step 4 ###########################################################
    # print('##########################################################################################################################\n')
    # print ('Step 4: check trends ')
    # print('\n##########################################################################################################################\n')
    # ######################################################################
    # #
    # df_MA = MA_datasets_SST.copy() # combine MA climate data and CRMS data
    # columns = ["Temp", "Salinity", "WL", "W_HP", "W_depth"]
    #
    # for col in columns:
    #     df_MA[col] = MA_datasets[col].median(axis=1, skipna=True)
    #     df_MA[col+"_Q1"] = MA_datasets[col].quantile(q=0.25, axis=1)
    #     df_MA[col+"_Q3"] = MA_datasets[col].quantile(q=0.75, axis=1)
    #
    # df_MA = df_MA[df_MA.index.notnull()]
    #
    # long_trends=calculate_trends(df_MA, slice(6)) # 1981 - 2022 for 5 variables
    # print('##########################################################################################################################\n')
    # print("\n\n","Long-term trends for climate driver is",long_trends)
    # print('\n##########################################################################################################################\n')
    #
    # # This is a plot data for paper
    # # Convert the datetime index to modified Julian date (number of days since November 17, 1858) For N_Graph but sligtly change the referenced days
    #
    # df_MA_temp = df_MA.copy().query('index <= @end_date')
    # print (df_MA_temp)
    # df_MA_temp.to_excel('MA_timeseris.xlsx')
    # df_MA_temp.fillna('=', inplace=True)
    # #df_MA.index = (df_MA.index - pd.Timestamp("1858-11-17")) // pd.Timedelta(days=1)
    # df_MA_temp.index = (df_MA_temp.index - pd.Timestamp(start_date_climate)) // pd.Timedelta(days=1)
    # output_name = 'MA_timeseris_plot.xlsx'
    # df_MA_temp.to_excel(output_name)
    # print(df_MA_temp.tail())
    #
    # df_MA=df_MA.query('index >= @start_date and index <= @end_date') # 2008 - 2022
    # df_MA_corr=df_MA.copy()
    # mid_trends=calculate_trends(df_MA, slice(None))
    # print('##########################################################################################################################\n')
    # print("\n\n","Mid-term trends (during observation ) is",mid_trends)
    # print('\n##########################################################################################################################\n')
    #
    # ### Step 5 ###########################################################
    # print('##########################################################################################################################\n')
    # print ('Step 5: check correlations ')
    # print('\n##########################################################################################################################\n')
    # ######################################################################
    #
    # corr=df_MA_corr.corr()
    # print ('The correlation of Moving average',corr)
    #
    # # Merge two datasets
    # for i in columns:
    #     SST[i] = np.nan # Create a column
    #     print(i)
    #     start_row = SST.index.get_loc(datasets[i].index[0])
    #     end_row = SST.index.get_loc(datasets[i].index[-1])
    #     print (start_row, end_row)
    #     SST.iloc[start_row:end_row+1,SST.columns.get_loc(i)] = datasets[i].median(axis=1,skipna=True) # each datasets have different length
    #
    # SST.head()
    # SST.to_excel('monthly_median_for_correlation.xlsx')
    # #
    # SST = SST.rename(columns={'SST': 'GoM SST',
    #                                       'Temp': 'CRMS ST','W_HP': 'HP','W_depth':'ID'})  # Change column name
    # SST_plot=SST.copy()
    # SST_plot.drop(['UV','ID'], axis=1, inplace=True)
    # SST_plot=SST_plot.query('index >= @start_date and index <= @end_date')
    # corr2=SST_plot.corr()
    # output = os.path.join(Photospace, 'correlation.png')
    # heatplot(corr2, output)
    #
    # ### Step 6 ###########################################################
    # print('##########################################################################################################################\n')
    # print ('Step 6: Analyze subdomain and vegetation')
    # print('\n##########################################################################################################################\n')
    # ######################################################################
    #
    # # The target grab point file
    # path_folder2 = "C:/Users/jikeda/Desktop/CRMS2Map/Code_dev/Input"
    # path_folder3 = "C:/Users/jikeda/Desktop/CRMS2Map/Code_dev/Output"
    # path_folder4 = "C:/Users/jikeda/Desktop/Time_Series_Analysis_version2/CRMS/CRMS_Marsh_Vegetation/"
    #
    # polygon_file=os.path.join(path_folder2,'Basin_NAD83.shp') # 10 basin provided by CPRA
    # Basin_community_file=os.path.join(path_folder4,'CRMS_station_Basin_Community.shp') # 5 vegetation (marsh) community analyzed by Jin
    #
    # Temp_point_file=os.path.join(path_folder3,'CRMS_stations_Water_Temp.shp')
    # Salinity_point_file=os.path.join(path_folder3,'CRMS_stations_Surface_salinity.shp')
    # W2l_point_file=os.path.join(path_folder3,'CRMS_stations_Water_Elevation_to_Datum.shp')
    # W2m_point_file=os.path.join(path_folder3,'CRMS_stations_Water_Elevation_to_Marsh.shp') # don't know why 'CRMS0287' is included in CRMS2plot code (12/19/23). Temporary ,manually delete 'CRMS0287'
    # #
    # ### 6.1 Open polygon and basin community files
    # polygon=gpd.read_file(polygon_file)
    # Basin_community=gpd.read_file(Basin_community_file)
    #
    # Temp_point = gpd.read_file(Temp_point_file)
    # Salinity_point=gpd.read_file(Salinity_point_file)
    # W2l_point = gpd.read_file(W2l_point_file)
    # W2m_point = gpd.read_file(W2m_point_file)
    # W2m_point = W2m_point[W2m_point.CRMS_Sta != 'CRMS0287']
    #
    # subset_file=[Temp_point,Salinity_point,W2l_point,W2m_point]
    # subset_file_name=["Temp","Salinity","WL","W2M"]
    #
    #
    # Subbasinspace = os.path.join(Workspace, 'Sub_basin')
    # Submarshspace = os.path.join(Workspace, 'Sub_marsh')
    # Subbasinspace_SLR = os.path.join(Workspace, 'Sub_basin_SLR')
    #
    # try:
    #     os.mkdir(Subbasinspace, exist_ok=True)
    #     os.mkdir(Submarshspace, exist_ok=True)
    #     os.mkdir(Subbasinspace_SLR, exist_ok=True)
    # except:
    #     pass
    #
    # # Convert to km2
    # polygon['Square_km']=polygon['ACRES']*0.00404686
    # print(polygon)
    #
    # # Check the long and lat
    # print(W2m_point)
    #
    # ### This is an optional on jupyterhub
    # import folium
    #
    # # base = explore_map(polygon, W2m_point,"station")
    # # base
    # # base2 = explore_map(polygon, Basin_community,"Community")
    # # base2
    #
    # print('##########################################################################################################################\n')
    # print('Grouped by sub_basin')
    # print('##########################################################################################################################\n')
    #
    # subsets = {} # add basin information on sub dataset
    # grouped_data = {} # grouped data in each subset
    #
    # for file, name in zip(subset_file, subset_file_name):
    #     print(name)
    #     try:
    #         subsets[name] = gpd.sjoin(file,polygon,how='inner',predicate='within')
    #         #output_subsets =name + "_subset.csv"
    #         #subsets[name].to_csv(output_subsets)
    #         grouped_data[name] = subsets[name].groupby('BASIN')
    #
    #     except UnicodeDecodeError:
    #         # If the above fails due to an encoding error, try another encoding
    #         print('Encoding error')
    #
    # basin_list=set(subsets["Temp"].BASIN.values)
    # print(basin_list) # check data
    # # print(grouped_data)
    #
    # # Create a index list for sub_basin dataset
    # basin_index, basin_key = create_subdomain_index(grouped_data, basin_list)
    # print (basin_index.keys())
    # print (basin_index['W2M']['BA']) # check the stations
    #
    # sort_basin=['PO','BS','MR','BA','TE','AT','TV','ME','CS'] # 'Perl basin is very small and exclude from analysis'
    #
    # MA_subsets = create_subdomain_datasets(datasets, file_name_o, basin_index, sort_basin, Subbasinspace, 'basin')
    #
    # #############################################
    # # Make median datasets for each domain and variable
    # #############################################
    # file1 = Subbasinspace +'/*_median.*csv'
    #
    # file_sub_name1=[]
    # file_sub_name_SLR=[]
    #
    # for sub_file in glob.glob(file1):
    #     if "_SLR_" not in sub_file:
    #         print(sub_file)
    #         file_sub_name1.append(sub_file)
    #     else:
    #         print(sub_file)
    #         file_sub_name_SLR.append(sub_file)
    #
    # file2 = Subbasinspace +'/*_median_MA.*csv'
    # file_sub_name2=[]
    #
    # for sub_file in glob.glob(file2):
    #     if "_SLR_" not in sub_file:
    #         print(sub_file)
    #         file_sub_name2.append(sub_file)
    #
    # sorted_file_name_o=['Salinity', 'Temp', 'WL', 'W_depth', 'W_HP'] # need to reorder the variables
    # print(file_sub_name1)
    #
    # subdatasets=sub_dataframe_gen(file_sub_name1,sorted_file_name_o)
    # subdatasets_SLR=sub_dataframe_gen(file_sub_name_SLR,file_name_o[-3:])
    # MA_subdatasets=sub_dataframe_gen(file_sub_name2,sorted_file_name_o)
    #
    # #############################################
    # # Display correlation plots
    # #############################################
    #
    # # corr={}
    # # for i in sorted_file_name_o:
    # #     print(i)
    # #     corr[i]=subdatasets[i].corr()
    # #
    # #     output = os.path.join(Photospace, 'sub_'+ i+ '.png')
    # #     heatplot(corr[i], output)
    #
    # #############################################
    # # Display basin level plots
    # #############################################
    #
    # # Temp
    #
    # plt.clf()
    # fig, ax = plt.subplots(figsize=(6,3))
    # ax.set_xlabel('Year')
    # ax.set_ylabel('$\it{T}$ [℃]')
    # lab=[]
    # for i,j in enumerate(sort_basin):
    #     ax.plot(MA_subdatasets["Temp"].index, MA_subdatasets["Temp"][j],color=color_palette[i],linewidth=1,label=j)
    # plt.xlim(mp_dates.date2num(plot_period))
    # plt.ylim([15, 30])
    # major_ticks = np.arange(15, 31, 5)
    # ax.set_yticks(major_ticks)
    # plt.grid(color = 'k', linestyle = '--', linewidth = 0.1)
    # B,C =ax.get_legend_handles_labels()
    # ax.legend(C,loc='lower right') # Rabeling
    #
    # B,C =ax.get_legend_handles_labels()
    # ax.legend(C,loc='lower right') # Rabeling
    # output = os.path.join(Photospace, 'Temp_basin.png')
    # plt.savefig(output,dpi=600,bbox_inches='tight')
    # #plt.show()
    # plt.close()
    #
    # # Salinity
    #
    # fig, ax = plt.subplots(figsize=(6,3))
    # ax.set_xlabel('Year')
    # ax.set_ylabel('$\it{S}$ [ppt]')
    # #
    # for i,j in enumerate(sort_basin):
    #     ax.plot(MA_subdatasets["Salinity"].index, MA_subdatasets["Salinity"][j],color=color_palette[i],linewidth=1,label=j)
    #
    # plt.xlim(mp_dates.date2num(plot_period))
    # plt.ylim([0, 20])
    # major_ticks = np.arange(0, 21, 4)
    # #minor_ticks = np.arange(0, 0.31, 1)
    # ax.set_yticks(major_ticks)
    # plt.grid(color = 'k', linestyle = '--', linewidth = 0.1)
    # B,C =ax.get_legend_handles_labels()
    # ax.legend(C,loc='upper right',ncol=2) # Rabeling
    # output = os.path.join(Photospace, 'Salinity_basin.png')
    # plt.savefig(output,dpi=600,bbox_inches='tight')
    # #plt.show()
    # plt.close()
    #
    # trends= calculate_trends(MA_subdatasets, "Salinity",subdomain=True, sort_domain_list=sort_basin)
    # print("\n\n","CRMS trends for climate driver is",trends)
    #
    # ##############################################################################################
    # # correlation with AR_Q
    #
    # correlations=[]
    # for col in MA_subdatasets["Salinity"].columns:  # Transpose to iterate over columns
    #     correlation = df_MA_corr.AR_Q.corr(MA_subdatasets["Salinity"][col])
    #     correlations.append(correlation)
    # print('The correlation between AR_Q and Salinity',correlations) # sort_basin=['PO','BS','MR','BA','TE','AT','TV','ME','CS']
    #
    # correlations=[]
    # for col in MA_subdatasets["WL"].columns:  # Transpose to iterate over columns
    #     correlation = df_MA_corr.AR_Q.corr(MA_subdatasets["WL"][col])
    #     correlations.append(correlation)
    # print('The correlation between AR_Q and Water level',correlations)
    # ##############################################################################################
    #
    # fig, ax = plt.subplots(figsize=(6,3))
    # ax.set_xlabel('Year')
    # ax.set_ylabel(r'$\xi$ [m,NAVD88]')
    #
    # for i,j in enumerate(sort_basin):
    #     ax.plot(MA_subdatasets["WL"].index, MA_subdatasets["WL"][j],color=color_palette[i],linewidth=1,label=j)
    #
    # plt.xlim(mp_dates.date2num(plot_period))
    # plt.ylim([0, 1.0])
    # major_ticks = np.arange(0, 1.1, 0.5)
    # #minor_ticks = np.arange(0, 0.31, 1)
    # ax.set_yticks(major_ticks)
    # plt.grid(color = 'k', linestyle = '--', linewidth = 0.1)
    # B,C =ax.get_legend_handles_labels()
    # ax.legend(C,loc='upper left',ncol=2) # Rabeling
    # output = os.path.join(Photospace, 'WL_basin.png')
    # plt.savefig(output,dpi=600,bbox_inches='tight')
    # #plt.show()
    # plt.close()
    #
    # trends= calculate_trends(MA_subdatasets, "WL",subdomain=True, sort_domain_list=sort_basin)
    # print("\n\n","CRMS trends for climate driver is",trends)
    #
    # #############################################
    # # Basin level precipitation
    # #############################################
    #
    # # file_name9="Basin_prcp"
    # file_name9="Basin_total_prcp" # Total precipitation have to update Jin 07/17/24
    # # file_name="CRMS_Surface_salinity_2006_2022"
    # file_name9=file_name9+file_suffix2
    #
    # Basin_prcp = create_dataframe(file_name9, "Date")
    #
    # # Calculate moving average
    # MA_datasets_Basin_prcp= Basin_prcp.rolling(window=MA_window, center=True).mean() # , min_periods=9min_period is a rough criterion
    #
    # #############################################
    # # Basin level correlation between CRMS and Climate driver
    # #############################################
    # SST_basin = SST.copy().query('index >= @start_date and index <= @end_date')
    # # need to update the SST_basin data to match the basin data
    # #corr_basin = create_subdomain_correlation(subdatasets, sorted_file_name_o, sort_basin, SST_basin, CS_Q, Basin_prcp, Subbasinspace, 'basin')
    #
    # #############################################
    # # Hydroperiod and inundation depth plots
    # #############################################
    #
    # # Create 5 years interval data
    # # Define the start and end years
    # start_year = int(start_date.split('-')[0])  # start_date
    # end_year = int(end_date.split('-')[0])  # end_date
    # interval = 5
    #
    # sorted_file_sub_name = get_sorted_files(Subbasinspace, 'W_HP_subset_*.csv', "W_HP_subset_basin_")
    # sorted_file_sub_name2 = get_sorted_files(Subbasinspace, 'W_depth_subset_*.csv', "W_depth_subset_basin_")
    # sorted_file_sub_name3 = get_sorted_files(Subbasinspace, 'W_HP_SLR_subset_*.csv', "W_HP_SLR_subset_basin_")
    # sorted_file_sub_name4 = get_sorted_files(Subbasinspace, 'W_depth_SLR_subset_*.csv', "W_depth_SLR_subset_basin_")
    #
    # datasets_HP = {}
    # datasets_HP.update(process_files(sorted_file_sub_name, sort_basin, "Now"))
    # datasets_HP.update(process_files(sorted_file_sub_name3, sort_basin, "SLR"))
    #
    # print(datasets_HP['Now']['PO']) # check the datasets
    #
    # HP_stats = pd.DataFrame()
    # # HP_stats = pd.DataFrame()
    #
    # for i in ["Now"]:
    #     aa = [] # dummy list
    #     for j in sort_basin:
    #         aa.append(datasets_HP[i][j].median())
    #     HP_stats[f"HP_{2008}"] = [item.iloc[2] for item in aa]
    #     HP_stats[f"Depth_{2008}"] = [item.iloc[3] for item in aa]
    #     HP_stats[f"HP_{2018}"] = [item.iloc[6] for item in aa]
    #     HP_stats[f"Depth_{2018}"] = [item.iloc[7] for item in aa]
    #
    # HP_stats.index = np.array(sort_basin).T
    # HP_stats['HP_diff'] = HP_stats.iloc[:, 2] - HP_stats.iloc[:, 0]
    # HP_stats['Depth_diff'] = HP_stats.iloc[:, 3] - HP_stats.iloc[:, 1]
    # HP_stats['HP_ratio'] = HP_stats.iloc[:, 2] / HP_stats.iloc[:, 0]
    # HP_stats['Depth_ratio'] = HP_stats.iloc[:, 3] / HP_stats.iloc[:, 1]
    # HP_stats.to_csv('HP_interval10.csv')
    #
    # # for i in ["Now", "SLR"]:
    # #     aa = []
    # #     for j in sort_basin:
    # #         aa.append(datasets_HP[i][j].median())
    # #     HP_stats_SLR[f"HP_{i}"] = [item.iloc[0] for item in aa]
    # #     HP_stats_SLR[f"Depth_{i}"] = [item.iloc[1] for item in aa]
    # #
    # # HP_stats_SLR.index=np.array(sort_basin).T
    # # HP_stats_SLR['HP_diff']=HP_stats_SLR.iloc[:,2]-HP_stats_SLR.iloc[:,0]
    # # HP_stats_SLR['Depth_diff']=HP_stats_SLR.iloc[:,3]-HP_stats_SLR.iloc[:,1]
    # # HP_stats_SLR['HP_ratio']=HP_stats_SLR.iloc[:,2]/HP_stats_SLR.iloc[:,0]
    # # HP_stats_SLR['Depth_ratio']=HP_stats_SLR.iloc[:,3]/HP_stats_SLR.iloc[:,1]
    # # HP_stats_SLR.to_csv('HP.csv')
    #
    # print(HP_stats)
    #
    # #######################################################################################################################
    # # plot
    # #######################################################################################################################
    # nested_data = datasets_HP["Now"]
    # period = ['Total', '08-12', '13-17', '18-22']
    # reshaped_data = []
    #
    # for location, variables in nested_data.items():
    #     print (location)
    #     col_list = np.arange(1, len(variables.columns), 2)
    #     col_nums = col_list.tolist()
    #
    #     HP = variables.iloc[:, ::2].copy()
    #     HP_merge = np.array(HP.values.flatten())
    #     print('Station num', int(HP_merge.size/4))
    #
    #     Depth = variables.iloc[:, col_nums].copy()
    #     Depth_merge = np.array(Depth.values.flatten())
    #
    #     periods = np.array(period * variables.shape[0]*col_list.size)
    #     locations = np.array([location] * variables.shape[0]*col_list.size)
    #
    #     # Append each row individually
    #     for loc, per, hp, depth in zip(locations, periods, HP_merge, Depth_merge):
    #         reshaped_data.append({'Basin': loc, 'Period': per, 'HP': hp, 'Depth': depth})
    #
    # df = pd.DataFrame(reshaped_data)
    # df.to_csv('check.csv')
    # df['Basin'] = pd.Categorical(df['Basin'], ["CS", "ME", "TV", "AT", "TE", "BA", "MR", "BS", "PO"])
    #
    # # Create a box plot using seaborn
    # plt.figure(figsize=(8, 5))
    # sns.set_theme(style='whitegrid',
    #               palette='Greys',  #hls
    #               font_scale=1)
    # # flierprops = dict(marker='o', markerfacecolor= 'none', markersize=1,
    # #                   linewidth=0, markeredgecolor='grey')
    #
    # ax=sns.boxplot(x='Basin', y='HP', hue='Period', data=df, showfliers=False,whis=0,linewidth=.5,medianprops={"linewidth": 2,
    #                         "solid_capstyle": "butt"})
    # ax.set_ylabel('$\it{HP}$')
    # ax.set_ylim([0, 1])  # Set y-axis limits
    # ax.yaxis.set_major_locator(MultipleLocator(0.2))
    # # plt.title('Box Plot of HP for Different Periods and Locations')
    # output = os.path.join(Photospace, 'HP_boxplot.png')
    # plt.savefig(output,dpi=600,bbox_inches='tight')
    # #plt.show()
    # plt.close()
    #
    # # Create a box plot using seaborn
    # plt.figure(figsize=(8, 5))
    # sns.set_theme(style='whitegrid',
    #               palette='Greys',  #hls
    #               font_scale=1)
    # # flierprops = dict(marker='o', markerfacecolor= 'none', markersize=1,
    # #                   linewidth=0, markeredgecolor='grey')
    #
    # ax=sns.boxplot(x='Basin', y='Depth', hue='Period', data=df, showfliers=False,whis=0,linewidth=.5,medianprops={"linewidth": 2,
    #                         "solid_capstyle": "butt"})
    # ax.set_ylabel('$\it{h}$')
    # ax.set_ylim([0, 0.5])  # Set y-axis limits
    # ax.yaxis.set_major_locator(MultipleLocator(0.2))
    # ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    # # plt.title('Box Plot of HP for Different Periods and Locations')
    # output = os.path.join(Photospace, 'Depth_boxplot.png')
    # plt.savefig(output,dpi=600,bbox_inches='tight')
    #
    # #plt.show()
    # plt.close()
    #
    #
    # # reset to the default settings
    # sns.reset_defaults()
    #
    #
    # print('##########################################################################################################################\n')
    # print('Grouped by vegetation')
    # print('##########################################################################################################################\n')
    #
    # # subset_file=[Temp_point,Salinity_point,W2l_point,W2m_point]
    # # subset_file_name=["Temp","Salinity","WL","W2M"]
    #
    # subsets_vegetation = {} # add basin information on sub dataset
    # grouped_data_vegetation = {} # grouped data in each subset
    #
    # for file, name in zip(subset_file, subset_file_name):
    #     print(name)
    #     try:
    #         #subsets_vegetation[name] = gpd.sjoin(file, polygon,how='inner',predicate='within')
    #
    #         df_merged = pd.merge(file, Basin_community[['Community', 'Count','Rate','BASIN', Basin_community.columns[1]]], how='left',
    #                      left_on=file.columns[0], right_on= Basin_community.columns[1])
    #         df_merged = df_merged.loc[~df_merged['Count'].isna()] # remove no basin datasets
    #         subsets_vegetation[name]=df_merged
    #         #output_subsets =name + "_vege_subset.csv"
    #         #subsets_vegetation[name].to_csv(output_subsets)
    #         grouped_data_vegetation[name] = subsets_vegetation[name].groupby('Community')
    #
    #     except UnicodeDecodeError:
    #         # If the above fails due to an encoding error, try another encoding
    #         print('Encoding error')
    #
    # vegetation_list=set(subsets_vegetation["Temp"].Community.values)
    #
    # # Create a index list for vegetation dataset
    # vegetation_index, vegetation_key = create_subdomain_index(grouped_data_vegetation, vegetation_list)
    # print(vegetation_index.keys())
    # print (vegetation_index['Temp'])
    #
    # sort_community=sorted(vegetation_list)
    # MA_subsets_vegetation = create_subdomain_datasets(datasets, file_name_o, vegetation_index, sort_community, Submarshspace, 'vegetation')
    #
    # #############################################
    # # Make median datasets for each vegetation and variable
    # #############################################
    # file1=Submarshspace +'/*_median.*csv'
    # file_sub_name1=[]
    #
    # for sub_file in glob.glob(file1):
    #     if "_SLR_" not in sub_file:
    #         print(sub_file)
    #         file_sub_name1.append(sub_file)
    #
    # file2=Submarshspace +'/*_median_MA.*csv'
    # file_sub_name2=[]
    #
    # for sub_file in glob.glob(file2):
    #     if "_SLR_" not in sub_file:
    #         print(sub_file)
    #         file_sub_name2.append(sub_file)
    #
    # sorted_file_name_o=['Salinity', 'Temp', 'WL', 'W_depth', 'W_HP'] # need to reorder the variables
    # print(sorted_file_name_o)
    #
    # subdatasets_vegetation=sub_dataframe_gen(file_sub_name1,sorted_file_name_o)
    # MA_subdatasets_vegetation=sub_dataframe_gen(file_sub_name2,sorted_file_name_o)
    #
    # #############################################
    # # Display correlation plots
    # #############################################
    # # corr_vegetation={}
    # # for i in sorted_file_name_o:
    # #     print(i)
    # #     corr_vegetation[i]=subdatasets_vegetation[i].corr()
    # #
    # #     output = os.path.join(Photospace,'sub_vegetation'+ i+ '.png')
    # #     heatplot(corr_vegetation[i], output)
    # #
    # # Temp
    #
    # fig, ax = plt.subplots(figsize=(6,3))
    # ax.set_xlabel('Year')
    # ax.set_ylabel(u'Temp [℃]')
    # lab=[]
    # for i,j in enumerate(sort_community):
    #     ax.plot(MA_subdatasets_vegetation["Temp"].index, MA_subdatasets_vegetation["Temp"][j],color=color_palette_vegetation[i],linewidth=1,label=j)
    # plt.xlim(mp_dates.date2num(plot_period))
    # plt.ylim([15, 30])
    # major_ticks = np.arange(15, 31, 5)
    # ax.set_yticks(major_ticks)
    # plt.grid(color = 'k', linestyle = '--', linewidth = 0.1)
    # B,C =ax.get_legend_handles_labels()
    # ax.legend(C,loc='lower right') # Rabeling
    #
    # B,C =ax.get_legend_handles_labels()
    # ax.legend(C,loc='lower right') # Rabeling
    # output = os.path.join(Photospace, 'Temp_marsh.png')
    # plt.savefig(output,dpi=600,bbox_inches='tight')
    # #plt.show()
    # plt.close()
    # #
    # # Salinity
    # #
    # fig, ax = plt.subplots(figsize=(6,3))
    # ax.set_xlabel('Year')
    # ax.set_ylabel('$\it{S}$ [ppt]')
    #
    # for i,j in enumerate(sort_community):
    #     ax.plot(MA_subdatasets_vegetation["Salinity"].index, MA_subdatasets_vegetation["Salinity"][j],color=color_palette_vegetation[i],linewidth=1,label=j)
    #
    # plt.xlim(mp_dates.date2num(plot_period))
    # plt.ylim([0, 20])
    # major_ticks = np.arange(0, 21, 4)
    # #minor_ticks = np.arange(0, 0.31, 1)
    # ax.set_yticks(major_ticks)
    # plt.grid(color = 'k', linestyle = '--', linewidth = 0.1)
    # B,C =ax.get_legend_handles_labels()
    # ax.legend(C,loc='upper right',ncol=2) # Rabeling
    # output = os.path.join(Photospace, 'Salinity_marsh.png')
    # plt.savefig(output,dpi=600,bbox_inches='tight')
    # #plt.show()
    # plt.close()
    #
    # trends= calculate_trends(MA_subdatasets_vegetation, "Salinity",subdomain=True, sort_domain_list=sort_community)
    # print("\n\n","CRMS trends for climate driver is",trends)
    #
    # print (MA_subdatasets_vegetation["Salinity"]["Freshwater"].mean(skipna=True))
    #
    # # WL
    # fig, ax = plt.subplots(figsize=(6,3))
    # ax.set_xlabel('Year')
    # ax.set_ylabel(r'$\xi$ [m,NAVD88]')
    #
    # for i,j in enumerate(sort_community):
    #     ax.plot(MA_subdatasets_vegetation["WL"].index, MA_subdatasets_vegetation["WL"][j],color=color_palette_vegetation[i],linewidth=1,label=j)
    # plt.xlim(mp_dates.date2num(plot_period))
    # plt.ylim([0, 1.0])
    # major_ticks = np.arange(0, 1.1, 0.5)
    # # minor_ticks = np.arange(0, 0.31, 1)
    # ax.set_yticks(major_ticks)
    # plt.grid(color='k', linestyle='--', linewidth=0.1)
    # B, C = ax.get_legend_handles_labels()
    # ax.legend(C, loc='upper left', ncol=2)  # labeling
    # output = os.path.join(Photospace, 'WL_marsh.png')
    # plt.savefig(output, dpi=600, bbox_inches='tight')
    # #plt.show()
    # plt.close()
    #
    # trends = calculate_trends(MA_subdatasets_vegetation, "WL", subdomain=True, sort_domain_list=sort_community)
    # print("\n\n", "CRMS trends for climate driver is", trends)
    #
    #
    # # Create a station precipitation data using basin prep
    # station_prcp = pd.DataFrame(columns=Basin_community['CRMS_Site'].values,index = Basin_prcp.index)
    #
    # for i,j in enumerate(station_prcp.columns):
    #     # print(j)
    #     community_basin=Basin_community['BASIN'][i]
    #     if community_basin == 'Pearl': # I don't create Pearl basin
    #         pass
    #     else:
    #         station_prcp[j]=Basin_prcp[community_basin]
    # print (station_prcp)
    #
    # # Calculate community prep
    # community_prcp = pd.DataFrame(columns=sort_community,index = Basin_prcp.index)
    #
    # grouped=Basin_community.groupby('Community')
    #
    # # Get the index of each group
    # #vegetation_index["Salinity"]
    #
    # # # Print the group indices
    # for group, indices in vegetation_index["Salinity"].items():
    #     #print(f"Group '{group}': {indices}")
    #     community_prcp[group]=station_prcp.loc[:,indices].mean(axis=1,skipna=True) # mean the values of each group
    # print (community_prcp)
    #
    # #############################################
    # # community level correlation between CRMS and Climate driver
    # #############################################
    # # Need to update the SST_basin data to match the basin data
    # #corr_community = create_subdomain_correlation(subdatasets_vegetation, sorted_file_name_o, sort_community, SST_basin, CS_Q, community_prcp, Submarshspace, 'community')
    #
    # print("Job Finished ʕ •ᴥ•ʔ")
    #
    # #For paper 2 row * 2 columns
    # sns.set_theme(style="whitegrid")
    # fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, figsize=(13, 7))
    # #fig.subplots_adjust(wspace=0.15, hspace=0.2)
    # fig.subplots_adjust(wspace=0.20, hspace=0.2)
    #
    # datasets_plot_lists = [MA_subdatasets_vegetation,MA_subdatasets_vegetation,MA_subdatasets, MA_subdatasets]
    # color_lists = [color_palette_vegetation,color_palette_vegetation,color_palette, color_palette]
    # y_variables = ['Salinity','WL','Salinity','WL']
    # types = [sort_community,sort_community,sort_basin, sort_basin]
    # legend_locs = ['upper center','upper center','upper center','upper left']
    # text_list = ['(A-1)', '(B-1)', '(A-2)', '(B-2)']
    # text_list = ['', '', '', '']
    #
    # for k, (dataset, color_list,y_variable, type,legend_loc) in enumerate(zip(datasets_plot_lists, color_lists, y_variables, types,legend_locs)):
    #     dataset_plot = dataset[y_variable]
    #     ax = axes[k//2, k%2]
    #     for i,j in enumerate(type):
    #
    #         ax.plot(dataset_plot.index, dataset_plot[j],color=color_list[i],linewidth=1,label=j)
    #
    #         if  k//2 == 1:
    #             ax.set_xlabel('Year')
    #
    #         if y_variable == 'Salinity':
    #             ax.set_ylabel('$\it{S}$ [ppt]')
    #             ax.set_ylim([0, 24])
    #             major_ticks = np.arange(0, 25, 4)
    #         elif y_variable == 'Temp':
    #             ax.set_ylabel(u'Temp [℃]')
    #         elif y_variable == 'WL':
    #             ax.set_ylabel(r'$\xi$ [m,NAVD88]')
    #             ax.set_ylim([0, 1.0])
    #             major_ticks = np.arange(0, 1.1, 0.5)
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
    # output = os.path.join(Photospace, 'Salinity_WL_combine.png')
    # plt.savefig(output,dpi=300,bbox_inches='tight')
    # plt.show()
    # plt.close()
    #
    # # # For paper 1 row * 2 columns
    # # sns.set_theme(style="whitegrid")
    # # fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, figsize=(12,4))
    # # fig.subplots_adjust(wspace=0.3, hspace=0.3)
    # #
    # # datasets_plot_lists = [MA_subdatasets_vegetation,MA_subdatasets_vegetation]
    # # color_lists = [color_palette_vegetation,color_palette_vegetation]
    # # y_variables = ['Salinity','WL']
    # # types = [sort_community,sort_community]
    # # legend_locs = ['upper right','upper right']
    # # text_list = ['(A)', '(B)']
    # #
    # # for k, (dataset, color_list,y_variable, type,legend_loc) in enumerate(zip(datasets_plot_lists, color_lists, y_variables, types,legend_locs)):
    # #     dataset_plot = dataset[y_variable]
    # #     ax = axes[k]
    # #     for i,j in enumerate(type):
    # #
    # #         ax.plot(dataset_plot.index, dataset_plot[j],color=color_list[i],linewidth=1,label=j)
    # #         ax.set_xlabel('Year')
    # #
    # #         if y_variable == 'Salinity':
    # #             ax.set_ylabel('$\it{S}$ [ppt]')
    # #             ax.set_ylim([0, 20])
    # #             major_ticks = np.arange(0, 21, 4)
    # #         elif y_variable == 'Temp':
    # #             ax.set_ylabel(u'Temp [℃]')
    # #         elif y_variable == 'WL':
    # #             ax.set_ylabel(r'$\xi$ [m,NAVD88]')
    # #             ax.set_ylim([0, 0.8])
    # #             major_ticks = np.arange(0, 0.9, 0.4)
    # #         elif y_variable == 'W_depth':
    # #             ax.set_ylabel('$\it{h}$')
    # #         elif y_variable == 'W_HP':
    # #             ax.set_ylabel('$\it{HP}$')
    # #         else:
    # #             pass
    # #
    # #     ax.set_xlim(mp_dates.date2num(plot_period))
    # #
    # #
    # #     ax.set_yticks(major_ticks)
    # #     ax.grid(color='k', linestyle='--', linewidth=0.1)
    # #     B,C =ax.get_legend_handles_labels()
    # #     ax.legend(C,loc=legend_loc ,ncol=2) # Rabeling
    # #
    # #     # Add the text from the text_list to the upper left corner of the plot
    # #     if text_list is not None and k < len(text_list):
    # #         ax.text(0.01, 0.98, text_list[k], transform=ax.transAxes, fontsize=12, weight='semibold',
    # #                 verticalalignment='top')
    # #
    # # output = os.path.join(Photospace, 'Salinity_WL_combine_single.png')
    # # plt.savefig(output,dpi=600,bbox_inches='tight')
    # # plt.show()
    # # plt.close()

    # Calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print("Done")
    print("Time to Compute: \t\t\t", elapsed_time, " seconds")
    print("Job Finished ʕ •ᴥ•ʔ")


if __name__ == "__main__":
    data_analysis()
