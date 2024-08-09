import os
import shutil
import pytest
import pandas as pd

from src.CRMS_general_functions import *
from src.CRMS_Continuous_Hydrographic2subsets import *

def test_download_CRMS(tmpdir):
    """Test the download_CRMS function with a sample URL."""
    url = "https://cims.coastal.la.gov/RequestedDownloads/ZippedFiles/CRMS_Discrete_Hydrographic.zip"  # Replace with a real URL if available
    zip_file = "CRMS_Discrete_Hydrographic.zip"
    csv_file = "CRMS_Discrete_Hydrographic.csv"
    input_space = tmpdir.mkdir("input")  # Create a temporary directory for the input space

    # Call the function to download the file
    result = download_CRMS(url, zip_file, csv_file, str(input_space))

def test_main_function(mocker, tmpdir):
    """Test the main function for CRMS_Continuous_Hydrographic2subsets.py."""
    original_input_space = os.path.join(os.getcwd(), "Input")

    # Create the temporary input directory within tmpdir
    tmp_input_space = tmpdir.mkdir("Input")

    # Copy necessary files from the original input space to tmpdir
    required_files = ["GEOID99_TO_GEOID12A.csv"]
    for file_name in required_files:
        src_file = os.path.join(original_input_space, file_name)
        dst_file = os.path.join(tmp_input_space, file_name)
        shutil.copy(src_file, dst_file)

    # Mock the os.getcwd() to return the temporary directory
    mocker.patch("os.getcwd", return_value=str(tmpdir))

    # Mock the download_CRMS function to avoid actual downloading
    mocker.patch("src.CRMS_general_functions.download_CRMS", return_value=True)

    # Call the function
    Continuous_Hydrographic()

    # Define the expected directories and files
    process_space = os.path.join(str(tmpdir), 'Process')

    # These are placeholders; replace with actual expected output files
    expected_files = [
        os.path.join(tmp_input_space, "GEOID99_TO_GEOID12A.csv")]

    # Check if the expected files were created
    for file_path in expected_files:
        assert os.path.exists(file_path), f"Expected file {file_path} does not exist."

    # Check if the Process directory was created
    assert os.path.exists(process_space), "Process directory was not created."

@pytest.fixture
def sample_data():
    """Fixture that returns a sample pandas DataFrame."""
    data = {
        'Station ID': ['CRMS1234-H01', 'CRMS1234-H02', 'CRMS5678-H01'],
        'Adjusted Salinity (ppt)': [2.5, 3.1, 4.0],
        'Adjusted Water Elevation to Marsh (ft)': [1.2, 1.3, 1.4],
        'Adjusted Water Elevation to Datum (ft)': [0.5, 0.6, 0.7],
        'Adjusted Water Temperature (°C)': [25.0, 26.5, 24.8],
        'Date': ['01/01/2020 00:00:00', '01/01/2020 00:01:00', '01/01/2020 00:02:00'],
        'Time Zone': ['CST', 'CST', 'CST']
    }
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df['Date'], format='%m/%d/%Y %H:%M:%S')
    return df


def test_pivot_data(sample_data):
    """Test that the pivot operation works as expected."""
    # Pivot the sample data
    pivoted_salinity = sample_data.pivot_table(index=sample_data.index, columns='Station ID',
                                               values='Adjusted Salinity (ppt)')

    # Check the shape of the pivoted DataFrame
    assert pivoted_salinity.shape == (3, 3), "Pivoted DataFrame does not have the expected shape"

    # Check specific values
    assert pivoted_salinity.loc['2020-01-01 00:00:00', 'CRMS1234-H01'] == 2.5, "Unexpected value in pivoted DataFrame"


def test_filter_and_clean_data(sample_data):
    """Test filtering and cleaning of data."""
    # Filter and clean data based on some criteria
    filtered_data = sample_data[sample_data['Station ID'].str.contains('-H')]
    filtered_data['Station ID'] = filtered_data['Station ID'].str.replace('-H\d+', '', regex=True)

    # Check that the filtering worked correctly
    assert filtered_data.shape[0] == 3, "Filtered data does not have the expected number of rows"
    assert all(filtered_data['Station ID'] == ['CRMS1234', 'CRMS1234', 'CRMS5678']), "Station ID cleaning failed"


def test_datetime_conversion(sample_data):
    """Test that datetime conversion and indexing works as expected."""
    sample_data.index.name = "Date"

    # Check if the index is correctly set to datetime
    assert pd.api.types.is_datetime64_any_dtype(sample_data.index), "Index is not datetime type"

    # Check if the datetime conversion is correct
    assert sample_data.index[0] == pd.Timestamp('2020-01-01 00:00:00'), "Datetime conversion failed"

