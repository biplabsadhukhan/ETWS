"""
===============================================================================
Storm–Tide Gauge Matching and Pseudo Stress Scaling for ETWS Analysis
===============================================================================

Description:
    This script combines the Extratropical Winter storms
    catalogue with high-pass filtered Sea Level Anomaly (SLA) observations
    from multiple NOAA tide gauge stations. For each storm, the script
    computes an pseudo-stress from pressure change and storm
    radius, matches hourly SLA observations to the storm timeline, and
    calculates storm-to-station distance and approach direction.

    The resulting dataset provides synchronized storm characteristics and
    coastal water level responses, enabling statistical analyses of ETWS
    impacts on extreme coastal water levels.

Features:
    - Loads ETWS storm catalogue
    - Reads high-pass filtered SLA data from multiple tide gauge stations
    - Matches SLA observations to each storm using timestamps
    - Estimates pseudo-stress from pressure gradient
    - Scales SLA using the pseudo-stress
    - Calculates storm-to-station distance
    - Computes storm bearing and compass direction
    - Exports the merged dataset in Parquet format

Requirements:
    - Python 3.x
    - pandas
    - numpy
    - xarray
    - scipy
    - geopy
    - matplotlib
    - pyarrow

Input:
    - ETWS storm catalogue (Parquet)
    - High-pass filtered Sea Level Anomaly (SLA) datasets
      for selected NOAA tide gauge stations

Output:
    A Parquet dataset containing:
        - Storm metadata
        - Storm intensity and radius
        - Estimated wind stress
        - Tide gauge SLA and sigma values
        - Stress-scaled SLA
        - Storm-to-station distance
        - Storm bearing and compass direction

Author:
    Dr. Biplab Sadhukhan
    April 07, 2025
    https://github.com/biplabsadhukhan


Notes:
    - Update all input and output directory paths before running the script.
    - Replace placeholder paths (e.g., "/path/to/input/" and
      "/path/to/output_directory/") with directories on your local system.
    - Modify the list of tide gauge stations if additional stations are
      included in the analysis.
    - The computed pseudo-stress is a simplified geostrophic estimate intended
      for statistical ETWS analysis rather than a full atmospheric forcing
      model.
===============================================================================
"""
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import math
#########################################################################
def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate bearing (azimuth) from (lat1, lon1) to (lat2, lon2).
    Result in degrees [0,360).
    """
    phi1, phi2 = map(math.radians, [lat1, lat2])
    dlon = math.radians(lon2 - lon1)

    x = math.sin(dlon) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)

    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360

def categorize_direction(bearing):
    """
    Convert numeric bearing into compass category.
    8 main directions (N, NE, E, SE, S, SW, W, NW).
    """
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    ix = round(bearing / 45) % 8
    return dirs[ix]

sla_files = [
    "/ExtraTropicalWinterStorms/Data/Eastport_ME_highpass.parquet",
    "/ExtraTropicalWinterStorms/Data/Portland_ME_highpass.parquet",
    "/ExtraTropicalWinterStorms/Data/Boston_MA_highpass.parquet",
    "/ExtraTropicalWinterStorms/Data/Newport_RI_highpass.parquet",
    "/ExtraTropicalWinterStorms/Data/The_Battery_NY_highpass.parquet",
    "/ExtraTropicalWinterStorms/Data/Cape_May_NJ_highpass.parquet",
    "/ExtraTropicalWinterStorms/Data/Lewes_DE_highpass.parquet",
    "/ExtraTropicalWinterStorms/Data/Beaufort_NC_highpass.parquet",
]

# Read each parquet and extract only required columns
dfs = []
for file in sla_files:
    df = pd.read_parquet(file, columns=["time", "sla", "sla_lowpass", "sla_highpass",
                                                        "sigma", "sla_lat", "sla_lon"])
    df["station"] = file.split("/")[-1].replace("_highpass.parquet", "")  # add station name for tracking
    dfs.append(df)

# Combine into one DataFrame
all_sla_data = pd.concat(dfs, ignore_index=True)
################################################################# Load land-sea mask dataset
# Load the Parquet file

input_path = "ExtraTropicalWinterStorms/Code/Data/coastal_diff_storm_data_1980-2024.parquet"


df = pd.read_parquet(input_path)
#################################################################
results = []

start_year, end_year = 1980,2024

df_time = df[(df['time'].dt.year >= start_year) & (df['time'].dt.year <= end_year)]


# ------------------------------------------------------------
# Map tide gauge stations to nearest coastal points
stations = {
    "Eastport_ME": (44.9, -67.0),
    "Portland_ME": (43.66, -70.25),
    "Boston_MA": (42.36, -71.05),
    "Newport_RI": (41.49, -71.32),
    "The_Battery_NY":(41.05,-71.96),
    "Cape_May_NJ": (38.97, -74.96),
    "Lewes_DE":(38.78,-75.12),
    "Beaufort_NC": (34.72, -76.67),
}


#################################################################################
final_dfs = []
# Constants
rho = 1025  # kg/m³, seawater density
Omega = 7.2921e-5 # rad/s

merged_results = []
for storm_id in df_time['id_year'].unique():
    print(f"Processing ID: {storm_id}")
    # Subset metadata for this storm
    df_meta = df[df['id_year'] == storm_id].set_index("time").copy()
    df_meta["id_year"] = storm_id

    # Compute Coriolis parameter
    df_meta["f"] = 2 * Omega * np.sin(np.deg2rad(df_meta["lat"]))

    # Compute speed magnitude = (1/(rho*f))*(intensity/radius)
    df_meta["speed_magnitude"] = (1 / (rho * df_meta["f"])) * (df_meta["intensity"] / df_meta["radius"])

    # Stress = speed_magnitude^2   (could also do rho_air * Cd * U^2 if desired)
    df_meta["stress"] = df_meta["speed_magnitude"]**2

    # Empty DF to hold per-station values
    df_stations = pd.DataFrame(index=df_meta.index)

    # Loop through tide gauge stations
    for station, (slat, slon) in stations.items():
        df_sla_station = all_sla_data[all_sla_data["station"] == station].copy()

        if "time" in df_sla_station.columns:
            df_sla_station["time"] = pd.to_datetime(df_sla_station["time"])
            # Align SLA/sigma to storm timeline
            df_sla_station = (
                df_sla_station.set_index("time")
                .reindex(df_meta.index, method="nearest"))

            if "sla" in df_sla_station.columns:
                df_stations[f"sla_{station}"] = df_sla_station["sla"].values
            if "sigma" in df_sla_station.columns:
                df_stations[f"sigma_{station}"] = df_sla_station["sigma"].values

    # Join storm metadata + station-specific data
    merged = df_meta.join(df_stations, how="inner")
    # Scale sigma by stress for each station
    for station in stations.keys():
        sla_col = f"sla_{station}"
        if sla_col in merged.columns:
            merged[f"sla_scaled_{station}"] = merged[sla_col] / merged["stress"]

    
    final_dfs.append(merged)

# Combine all storms into one DF
final_df = pd.concat(final_dfs).reset_index()

for station, (s_lat, s_lon) in stations.items():
    # Distance from tidegauge
    colname_dist = f"distance_{station}"
    final_df[colname_dist] = final_df.apply(lambda row: geodesic((row["lat"], row["lon"]), (s_lat, s_lon)).km,
        axis=1)

    # Bearing
    colname_dir = f"direction_{station}"
    final_df[colname_dir] = final_df.apply(lambda row: calculate_bearing(s_lat, s_lon, row["lat"], row["lon"]),
        axis=1)

    # Compass
    colname_compass = f"compass_{station}"
    final_df[colname_compass] = final_df[colname_dir].apply(categorize_direction)

output_dir = "ExtraTropicalWinterStorms/Data/"
final_df.to_parquet(f"{output_dir}Storms_radius_stress_distance_all_tidegauges_{start_year}-{end_year}.parquet", index=False)

