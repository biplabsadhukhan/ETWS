
"""
===============================================================================
Tide Gauge Detiding Using UTide Harmonic Analysis
===============================================================================

Description:
    This script removes the astronomical tidal signal from hourly NOAA CO-OPS
    tide gauge water level observations using the UTide harmonic analysis
    package. The resulting detided water level represents Sea Level Anomaly
    (SLA), which is suitable for storm surge and extreme water level analyses.

    The script processes all selected tide gauge records, checks data coverage,
    performs harmonic tidal analysis, reconstructs the predicted tide, and
    subtracts it from the observed water level to generate SLA. The processed
    datasets are saved as individual Apache Parquet files.

Features:
    - Reads hourly tide gauge data in Parquet format
    - Verifies sufficient temporal coverage (1980–2024)
    - Performs harmonic tidal analysis using UTide
    - Computes Sea Level Anomaly (SLA)
    - Preserves station metadata (latitude, longitude, station name)
    - Saves detided datasets in compressed Parquet format
    - Handles stations where harmonic analysis fails

Requirements:
    - Python 3.x
    - pandas
    - scipy
    - utide
    - pyarrow

Input:
    Hourly NOAA CO-OPS tide gauge water level data (.parquet/.prq.gzip)

Output:
    One Parquet file per station containing:
        - date_time
        - Sea Level Anomaly (SLA)
        - latitude
        - longitude
        - tide gauge name

Author:
    Dr.Biplab Sadhukhan
    Dec 26, 18:53:18 2024
    https://github.com/biplabsadhukhan
    

Notes:
    - Update the input and output directory paths before running.
    - Stations with insufficient temporal coverage or unsuccessful UTide
      harmonic fits are skipped automatically.
    - The generated Sea Level Anomaly (SLA) data are intended for storm surge,
      extreme water level, and coastal hazard analyses.
===============================================================================
"""
import pandas as pd
import glob
import os
import utide
from scipy.linalg import LinAlgError
######## Directory of the tidegauges change according to your path
dir_path = "selected_tidegauges/*.prq.gzip"
file_list = glob.glob(dir_path)

start_date = pd.to_datetime('1980-01-01 00:00:00')
end_date = pd.to_datetime('2024-12-31 23:00:00')

for file in file_list:
    try:
        file_name = os.path.basename(file)
        tidegauge_name = file_name.split('.')[0]
        data = pd.read_parquet(file)
        data['date_time'] = pd.to_datetime(data['date_time'])

        # Filter the time window
        filtered = data[(data['date_time'] >= start_date) & (data['date_time'] <= end_date)]

        # Check for adequate coverage
        if filtered['date_time'].min() > start_date or filtered['date_time'].max() < end_date:
            print(f"{tidegauge_name}: Data not in sufficient range for detiding (needs full 1980–2024).")
            continue
        
        # Detide the data
        toDetide = ['value']
        detided = pd.DataFrame()
        for k in toDetide:
            tideFit = utide.solve(data['date_time'], data[k], lat=data['lat'][0], method="ols", conf_int="MC", verbose=False)
            tidePredict = utide.reconstruct(data['date_time'], tideFit, verbose=False)
            detided['SLA'] = data[k] - tidePredict.h  # Rename 'value' to 'SLA'

        # Add latitude, longitude, date, and tide gauge name to the detided data
        detided['lat'] = data['lat'].round(2)
        detided['lon'] = data['lon'].abs().round(2)
        detided['date_time'] = data['date_time']
        detided['tide_gauge'] = tidegauge_name

        detided.to_parquet(f"parquet_detited/{tidegauge_name}.parquet")

        print(f"{tidegauge_name} detided data saved successfully!")

    except LinAlgError:
        print(f"{tidegauge_name}: UTide fit did not converge. Please inspect data quality.\n")

print("Tide gauge data processing complete!")
