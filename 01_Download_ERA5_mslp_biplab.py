
"""
===============================================================================
ERA5 Mean Sea Level Pressure (MSLP) Download Script
===============================================================================

Description:
    This script downloads hourly ERA5 Mean Sea Level Pressure (MSLP) data
    from the Copernicus Climate Data Store (CDS) using the CDS API.
    The current configuration retrieves data for the Bay of Bengal region
    (30°N–2°N, 75°E–110°E) for the years 2000–2001 and saves the output
    in NetCDF format.

Requirements:
    - Python 3.x
    - cdsapi package
    - Active Copernicus Climate Data Store (CDS) account
    - Valid CDS API credentials configured on your system

Authentication:
    Before running this script:
      1. Create a free CDS account:
         https://cds.climate.copernicus.eu/
      2. Obtain your CDS API credentials.
      3. Configure the CDS API credentials according to the official
         CDS API documentation.

Output:
    - Dataset: ERA5 Single Levels
    - Variable: Mean Sea Level Pressure (MSLP)
    - Temporal Resolution: Hourly
    - Format: NetCDF

Author:
    Dr.Biplab Sadhukhan
    September 19, 2024
    https://github.com/biplabsadhukhan

===============================================================================
"""
#----------------------------------------------------------------------------------------#
import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": ["mean_sea_level_pressure"],
    "year": [
        "1980", "2024"
    ],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [30, 75, 2, 110]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download('mslp/ERA_bob_mslp_1980_2024.nc')