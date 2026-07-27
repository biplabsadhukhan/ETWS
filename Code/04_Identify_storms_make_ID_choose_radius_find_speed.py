"""
===============================================================================
Extratropical Storm (ETWS) Identification and Tracking from ERA5 MSLP
===============================================================================

Description:
    This script identifies and tracks Extratropical Weather Systems (ETWS)
    using hourly ERA5 Mean Sea Level Pressure (MSLP) data. Storm centers are
    detected as local pressure minima below a specified pressure threshold and
    linked through time to construct storm tracks.

    For each storm, the script computes several characteristics including:
        - Storm center location (latitude and longitude)
        - Central pressure
        - Storm intensity (pressure change)
        - Storm radius
        - Storm duration
        - Storm track length
        - Mean translation speed
        - Storm movement direction

    Storms are filtered based on minimum track length and geographic
    constraints, and only storms passing through the specified study region are
    retained. The final storm catalogue is saved in Apache Parquet.

Features:
    - Detects storm centers from ERA5 MSLP fields
    - Tracks storms through consecutive hourly time steps
    - Calculates storm intensity and radius
    - Computes storm duration, path length, and propagation speed
    - Determines storm movement direction (bearing and compass direction)
    - Filters storms by geographic region and minimum track length
    - Exports a complete storm catalogue in Parquet format

Requirements:
    - Python 3.x
    - xarray
    - numpy
    - pandas
    - scipy
    - geopy

Input:
    - ERA5 Mean Sea Level Pressure (MSLP) NetCDF files
    - ERA5 Land-Sea Mask NetCDF file

Output:
    A Parquet file containing the storm catalogue with:
        - Storm ID
        - Date and time
        - Storm center latitude and longitude
        - Central pressure
        - Storm intensity
        - Storm radius
        - Storm duration
        - Storm track length
        - Storm translation speed

Author:
    Dr. Biplab Sadhukhan
    March 26, 2025
    https://github.com/biplabsadhukhan

Notes:
    - Update all input and output directory paths to match your local system.
    - Replace placeholder paths (e.g., "/path/to/output_directory/") with your
      preferred directories before running the script.
    - The pressure threshold, tracking distance, and study domain can be
      modified to suit different storm detection applications.
===============================================================================
"""
import xarray as xr
import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter, label
from collections import defaultdict
from geopy.distance import geodesic
import math
from scipy.ndimage import gaussian_filter1d
#######################################################################################################
# Function to find storm centers

def find_storm_centers(slp, threshold=1):
    local_min = minimum_filter(slp, size=3)
    storm_centers = np.isclose(slp, local_min) & (slp < np.roll(slp, 1, axis=0)) & (slp < np.roll(slp, -1, axis=0))
    storm_centers = storm_centers & (slp < threshold)

    # Remove border artifacts
    storm_centers[:1, :] = storm_centers[-1:, :] = False
    storm_centers[:, :1] = storm_centers[:, -1:] = False

    labeled_array, num_features = label(storm_centers)
    centers = [np.column_stack(np.where(labeled_array == i))[0] for i in range(1, num_features + 1)]
    return centers
    
def classify_point(lon, lat, mask):
    # Find nearest longitude and latitude indices
    lon_idx = np.argmin(np.abs(mask.coords["longitude"].values - lon))
    lat_idx = np.argmin(np.abs(mask.coords["latitude"].values - lat))
    classification = mask.isel(latitude=lat_idx, longitude=lon_idx).values
    classification_scalar = float(classification) if classification.size == 1 else classification.item()
    return "Land" if classification_scalar >= 0.5 else "Sea"
# Calculate the bearing (direction) between two locations
def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    delta_lon = lon2_rad - lon1_rad
    x = math.sin(delta_lon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)

    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360  # Normalize to 0-360 degrees
    return compass_bearing
def bearing_to_compass(bearing):
    compass_points = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    index = round(bearing / 45) % 8
    return compass_points[index]

# Function to generate cross-sectional pressure
def cross_section_pressure_simple(slp, storm_center, axis='longitude'):
    center_y, center_x = storm_center
    if axis == 'longitude':
        cross_section = slp[center_y, :]
        distances = np.arange(len(cross_section))
        #distances = np.arange(0, len(cross_section)) 
        xCenter_exp = center_x  # Define xCenter_exp as the storm center along longitude
    elif axis == 'latitude':
        cross_section = slp[:, center_x]
        distances = np.arange(len(cross_section))
        xCenter_exp = center_y  # Define xCenter_exp as the storm center along latitude
    else:
        raise ValueError("Axis must be either 'longitude' or 'latitude'.")
    return distances, cross_section, xCenter_exp
##############################################################################
# Load SLP datasets
ds1 = xr.open_dataset('home/biplab/ERA_data/Sea_Pressure/ERA_1980_1989_mslp.nc')
ds2 = xr.open_dataset('home/biplab/ERA_data/Sea_Pressure/ERA_1990_1999_mslp.nc')
ds3 = xr.open_dataset('home/biplab/ERA_data/Sea_Pressure/ERA_2000_2007_mslp.nc')
ds4 = xr.open_dataset('home/biplab/ERA_data/Sea_Pressure/ERA_2008_2015_mslp.nc')
ds5 = xr.open_dataset('home/biplab/ERA_data/Sea_Pressure/ERA_2016_2024_mslp.nc')

ds_list = [ds.drop_vars("expver", errors="ignore") for ds in [ds1, ds2, ds3, ds4, ds5]]
ds_aligned = xr.align(*ds_list, join="inner", exclude=["valid_time"])
ds = xr.concat(ds_aligned, dim="valid_time")
ds['msl'] = ds['msl'] / 100.0  # Convert Pa to hPa
ds = ds.sel(
    valid_time=((ds.valid_time.dt.month >= 12) | (ds.valid_time.dt.month <= 3)),
    latitude=slice(49, 25),
    longitude=slice(-90, -58))

################################################################# Load land-sea mask dataset
ds_mask = xr.open_dataset("home/biplab/ERA_data/Land/ERA_land.nc")
land_sea_mask = ds_mask["lsm"].isel(valid_time=0)
#####################################################################
storm_data = []
start_year, end_year = 1980, 2024

for year in range(start_year, end_year + 1):
    print(f"Processing year: {year}")

    storm_tracks = defaultdict(list)
    storm_id = 0

    ds_year = ds.sel(valid_time=ds.valid_time.dt.year == year)

    for time in ds_year.valid_time:
        slp = ds_year.sel(valid_time=time)['msl'].values
        centers = find_storm_centers(slp, threshold=1000)
        
        if centers:
            min_p = float('inf')
            best_center = None
            for c in centers:
                p = slp[c[0], c[1]]
                if p < min_p:
                    min_p = p
                    best_center = c

            center_lat = float(ds_year.latitude[best_center[0]].values)
            center_lon = float(ds_year.longitude[best_center[1]].values)
            P_center = round(float(min_p), 2)

            # Cross-section through the min center
            distances, pressure_values, xCenter_exp = cross_section_pressure_simple(slp, best_center, axis='longitude')
            

            #pressure_values = gaussian_filter1d(pressure_values1, sigma=2)
            #smoothed_left_pressures = gaussian_filter1d(left_pressures, sigma=2)

            
            left_mask = distances < xCenter_exp  # Consider only the left side of the curve
            left_distances = distances[left_mask]
            left_pressures = pressure_values[left_mask]
            
            # Filter pressures within ranges
            valid_mask_1 = (left_pressures > 1000) & (left_pressures <= 1040)
            valid_pressures_1 = left_pressures[valid_mask_1]
            valid_distances_1 = left_distances[valid_mask_1]
   
    
            # Initialize values
            pressure_max, radius_selected= None, None

            # Determine which range contains valid pressures
            for valid_pressures, valid_distances in [(valid_pressures_1, valid_distances_1)]:
                if valid_pressures.size > 0:
                    pressure_max_index = np.argmax(valid_pressures)
                    pressure_max = valid_pressures[pressure_max_index]
                    radius_selected = abs(valid_distances[pressure_max_index] - xCenter_exp)
                    break
            
            if pressure_max is not None:
                    intensity_selected = pressure_max - P_center
           
           
            assigned = False
            for sid, track in storm_tracks.items():
                last_time, last_lat, last_lon, *_ = track[-1]
                time_diff = (np.datetime64(time.values) - np.datetime64(last_time)).astype('timedelta64[h]').astype(int)
                d_storm = np.sqrt((center_lat - last_lat)**2 + (center_lon - last_lon)**2)
                if time_diff == 1 and d_storm < 1.1:
                    storm_tracks[sid].append((time.values, center_lat, center_lon, intensity_selected,P_center, radius_selected))
                    assigned = True
                    break
            if not assigned:
                storm_id += 1
                storm_tracks[storm_id].append((time.values, center_lat, center_lon,intensity_selected, P_center, radius_selected))
        

    # Filter and renumber storms for current year
    storm_tracks = {sid: track for sid, track in storm_tracks.items() if len(track) > 5}
    renumbered_storm_tracks = {}
    for new_sid, (old_sid, track) in enumerate(storm_tracks.items(), start=1):
        renumbered_storm_tracks[new_sid] = track
        
    def track_land_count(track):
        """
        track: list of tuples like (time, center_lat, center_lon, P_center, radius_selected)
        returns number of centers classified as 'Land'
        """
        
    # Store final storm data
    for new_sid, track in renumbered_storm_tracks.items():
        for entry in track:
            time, center_lat, center_lon,intensity_selected,P_center,radius_selected = entry
            id_year = year * 1000 + new_sid
            lsm = classify_point(center_lon, center_lat, land_sea_mask)
            radius_selected1=radius_selected*0.25
            radius_km = radius_selected1 * 111.32 * np.cos(np.radians(center_lat))
            storm_data.append([id_year, new_sid, time, center_lat, center_lon,intensity_selected, P_center,radius_km,radius_selected1])

df = pd.DataFrame(storm_data,columns=["id_year", "storm_id", "time", "lat", "lon","intensity","central_pressure", "radius","unit_radius"])
#----------------------------------------------Storms with radius, intensity and ID created------------------#
#Length of storm, Speed of storm 
def compute_storm_metrics(track_df):
    track_df = track_df.sort_values("time")
    coords = list(zip(track_df["lat"], track_df["lon"]))

    total_length = sum(
        geodesic((lat1, lon1), (lat2, lon2)).kilometers
        for (lat1, lon1), (lat2, lon2) in zip(coords[:-1], coords[1:]))

    dt = (track_df["time"].iloc[1] - track_df["time"].iloc[0]) / np.timedelta64(1, "h")
    duration_h = (track_df["time"].iloc[-1] - track_df["time"].iloc[0]) / np.timedelta64(1, "h") + dt


    duration_s = duration_h * 3600
    speed_ms = (total_length * 1000) / duration_s if duration_s > 0 else np.nan
  

    return pd.Series({
        "storm_length_km": total_length,
        "storm_duration_h": duration_h,
        "storm_speed_ms": speed_ms})
    speed_ms = (total_length * 1000) / duration_s if duration_s > 0 else np.nan



storm_metrics = df.groupby("id_year").apply(compute_storm_metrics).reset_index()
df = df.merge(storm_metrics, on="id_year", how="left")

# Define bounding box
lon_min, lon_max = -82, -67
lat_min, lat_max = 34, 45

# Select storms with at least one center inside the box
df_box = df[(df["lon"] >= lon_min) & (df["lon"] <= lon_max) &
            (df["lat"] >= lat_min) & (df["lat"] <= lat_max)]
# Get unique storm IDs (id_year) that passed through the box
selected_ids = df_box["id_year"].unique()


all_storm_ids = df["id_year"].unique()


not_considered_ids = np.setdiff1d(all_storm_ids, selected_ids)

# --- Create DataFrame of non-considered storms ---
df_not_considered = df[df["id_year"].isin(not_considered_ids)].copy()

df_select = df[df["id_year"].isin(selected_ids)] 
print(f"Number of storms passing through box: {len(selected_ids)}")

# --- Compute net displacement direction per storm ---
def storm_direction_classification(track_df):
    track_df = track_df.sort_values("time")
    start_lat, start_lon = track_df.iloc[0][["lat", "lon"]]
    end_lat, end_lon = track_df.iloc[-1][["lat", "lon"]]

    bearing = calculate_bearing(start_lat, start_lon, end_lat, end_lon)
    compass = bearing_to_compass(bearing)

    return pd.Series({
        "start_lat": start_lat,
        "end_lat": end_lat,
        "start_lon": start_lon,
        "end_lon": end_lon,
        "bearing": bearing,
        "compass": compass
    })


df_new = df_select[df_select["id_year"].isin(selected_ids)].copy()

df_new["year"] = df_new["id_year"] // 1000
storm_first_time = df_new.groupby(["year", "storm_id"])["time"].min().reset_index() # Find the first time each storm appears
storm_first_time = storm_first_time.sort_values(["year", "time"]) # Sort storms within each year by first appearance
storm_first_time["new_id"] = storm_first_time.groupby("year").cumcount() + 1 # Assign new storm_id starting from 1 within each year

df_new = df_new.merge(storm_first_time[["year", "storm_id", "new_id"]],
                        on=["year", "storm_id"],
                        how="left")

df_new["storm_id"] = df_new["new_id"] # Replace storm_id with new_id

# Recalculate id_year
df_new["id_year"] = df_new["year"] * 1000 + df_new["storm_id"]

# Drop helper columns
df_new = df_new.drop(columns=["year", "new_id"])

# Remove them from both DataFrames

print(f"Number of storms NOT considered: {len(df_not_considered['id_year'].unique())}")
print(f"Number of storms selected: {len(df_new['id_year'].unique())}")

# #specify where the files that will end up in ETWS are located
output_dir = "ExtraTropicalWinterStorms/Data/"

df_new.to_parquet(f"{output_dir}coastal_diff_storm_data_{start_year}-{end_year}_intensity.parquet", index=False)




