
"""
===============================================================================
Extratropical Weather System (ETWS) Identification and Tracking Using
Double-Exponential Pressure Profile Fitting
===============================================================================

Description:
    This script identifies and tracks Extratropical Weather Systems (ETWS)
    from hourly ERA5 Mean Sea Level Pressure (MSLP) fields using an enhanced
    storm detection methodology. Storm centers are detected as local pressure
    minima, and a double-exponential pressure profile is fitted to the
    pressure cross-section through each storm center to estimate storm
    structure.

    Unlike the standard threshold-based method, storm radius is determined
    from the fitted pressure profile by locating the pressure threshold
    crossing (e.g., 1010 hPa) using interpolation. If no threshold crossing
    exists, a physically based half-decay estimate derived from the fitted
    exponential profile is used as a fallback. Storm intensity is computed
    from the pressure difference between the storm center and the fitted
    pressure at the selected radius.


Features:
    - Detects storm centers from ERA5 MSLP fields
    - Fits a double-exponential pressure profile to each storm
    - Estimates storm radius using pressure-threshold interpolation
    - Uses a half-decay fallback radius when no threshold crossing exists
    - Computes storm intensity from the fitted pressure profile
    - Tracks storms through consecutive hourly time steps
    - Filters storms based on track length and geographic criteria
    - Exports the final storm catalogue in Parquet format

Requirements:
    - Python 3.x
    - xarray
    - numpy
    - pandas
    - scipy
    - geopy
    - pyarrow

Input:
    - ERA5 Mean Sea Level Pressure (MSLP) NetCDF files
    - ERA5 Land-Sea Mask dataset

Output:
    A Parquet storm catalogue containing:
        - Storm ID
        - Date and time
        - Storm center latitude and longitude
        - Central pressure
        - Storm intensity
        - Fitted storm radius
        - Storm duration
        - Storm track length
        - Storm translation speed
        - Storm movement direction

Author:
    Dr. Biplab Sadhukhan
    March 29, 2026
    https://github.com/biplabsadhukhan

Notes:
    - Update all input and output directory paths before running the script.
    - Replace placeholder paths (e.g., "/path/to/input/" and
      "/path/to/output_directory/") with directories on your local system.
    - The pressure threshold, fitting parameters, and storm tracking criteria
      can be modified for different ETWS applications.
    - This implementation provides an alternative storm-radius estimation
      technique based on fitted pressure profiles rather than direct pressure
      gradients.
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
from scipy.optimize import minimize
##########################################################################################
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
# Define double exponent function for curve fitting
def doubleExponent(P, x):
    lamb, xCenter, C, A, B = P
    return C * np.exp(-lamb * np.abs(x - xCenter)) + A * x + B

# Define cost function for optimization
def costFunction(P, x, yTrue, curveFunc):
    return np.sum((yTrue - curveFunc(P, x))**2.0)

##############################################################################
# Load SLP datasets
ds1 = xr.open_dataset('ERA_data/Sea_Pressure/ERA_1980_1989_mslp.nc')
ds2 = xr.open_dataset('ERA_data/Sea_Pressure/ERA_1990_1999_mslp.nc')
ds3 = xr.open_dataset('ERA_data/Sea_Pressure/ERA_2000_2007_mslp.nc')
ds4 = xr.open_dataset('ERA_data/Sea_Pressure/ERA_2008_2015_mslp.nc')
ds5 = xr.open_dataset('ERA_data/Sea_Pressure/ERA_2016_2024_mslp.nc')

ds_list = [ds.drop_vars("expver", errors="ignore") for ds in [ds1, ds2, ds3, ds4, ds5]]
ds_aligned = xr.align(*ds_list, join="inner", exclude=["valid_time"])
ds = xr.concat(ds_aligned, dim="valid_time")
ds['msl'] = ds['msl'] / 100.0  # Convert Pa to hPa
ds = ds.sel(
    valid_time=((ds.valid_time.dt.month >= 12) | (ds.valid_time.dt.month <= 3)),
    latitude=slice(45, 25),
    longitude=slice(-90, -58))

################################################################# Load land-sea mask dataset
ds_mask = xr.open_dataset("/home/biplab/ERA_data/Land/ERA_land.nc")
land_sea_mask = ds_mask["lsm"].isel(valid_time=0)
#####################################################################
storm_data = []
start_year, end_year = 2000, 2024

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
            distances, pressure_values1, xCenter_exp = cross_section_pressure_simple(slp, best_center, axis='longitude')
            

            pressure_values = gaussian_filter1d(pressure_values1, sigma=2)
            
            
            # --- improved initial guesses and bounds for fitting ---
            Pinit_exp = [0.05, float(xCenter_exp), -(np.max(pressure_values) - np.min(pressure_values)), 0.0, np.median(pressure_values)]
            bounds = [
                (1e-4, 1.0),                                # lamb
                (0.0, float(len(distances)-1)),             # xCenter
                (-(np.max(pressure_values) - np.min(pressure_values))*5, 0.0),  # C negative amplitude
                (-5.0, 5.0),                                # A slope
                (np.min(pressure_values)-20, np.max(pressure_values)+20)       # B baseline
            ]

            try:
                result_exp = minimize(costFunction, Pinit_exp, args=(distances, pressure_values, doubleExponent),
                                      method='L-BFGS-B', bounds=bounds, options={'maxiter':2000})
            except Exception:
                result_exp = minimize(costFunction, Pinit_exp, args=(distances, pressure_values, doubleExponent),
                                      method='Nelder-Mead', options={'maxiter':2000})

            Pestimate_exp = result_exp.x
            yEst_exp = doubleExponent(Pestimate_exp, distances)
            lamb, xCenter_fit, C, A, B = Pestimate_exp
            cost_exp = costFunction(Pestimate_exp, distances, pressure_values, doubleExponent)

            # ===== robust threshold-based radius with interpolation =====
            def get_radius_from_threshold(distances, fitted_curve, center_idx=None, threshold=1010):
                """
                Returns (radius, index_at_radius, side) where index_at_radius is the index in distances
                corresponding to the threshold crossing and side is 'left' or 'right'.
                If no crossing found, returns (None, None, None).
                """
                if center_idx is None:
                    center_idx = int(np.argmin(fitted_curve))

                # LEFT: search from center down to index 0
                left_x = distances[:center_idx+1]
                left_y = fitted_curve[:center_idx+1]
                left_cross_idx = None
                for i in range(len(left_y)-1, 0, -1):
                    # we're looking for transition from <threshold (near center) to >=threshold (farther out)
                    if left_y[i] < threshold and left_y[i-1] >= threshold:
                        # linear interpolation between i and i-1
                        x1, y1 = left_x[i], left_y[i]
                        x0, y0 = left_x[i-1], left_y[i-1]
                        frac = (threshold - y1) / (y0 - y1) if (y0 - y1) != 0 else 0.0
                        left_cross = x1 + frac * (x0 - x1)
                        left_cross_idx = (i, i-1, left_cross)  # keep for reference
                        break
                    if left_y[i-1] == threshold:
                        left_cross = left_x[i-1]
                        left_cross_idx = (i-1, i-1, left_cross)
                        break

                # RIGHT: search from center to end
                right_x = distances[center_idx:]
                right_y = fitted_curve[center_idx:]
                right_cross_idx = None
                for i in range(0, len(right_y)-1):
                    if right_y[i] < threshold and right_y[i+1] >= threshold:
                        x1, y1 = right_x[i], right_y[i]
                        x2, y2 = right_x[i+1], right_y[i+1]
                        frac = (threshold - y1) / (y2 - y1) if (y2 - y1) != 0 else 0.0
                        right_cross = x1 + frac * (x2 - x1)
                        # convert to absolute index in distances
                        abs_idx = center_idx + i
                        right_cross_idx = (abs_idx, abs_idx+1, right_cross)
                        break
                    if right_y[i+1] == threshold:
                        right_cross = right_x[i+1]
                        right_cross_idx = (center_idx + i + 1, center_idx + i + 1, right_cross)
                        break

                # compute radii from center
                left_radius = None
                left_idx_out = None
                if left_cross_idx is not None:
                    # left_cross_idx[2] holds interpolated distance (in same units as distances)
                    left_radius = center_idx - left_cross_idx[2]
                    # pick nearest integer index for intensity lookup (use the farther grid point index)
                    left_idx_out = left_cross_idx[0]  # nearer to center index (i)
                right_radius = None
                right_idx_out = None
                if right_cross_idx is not None:
                    right_radius = right_cross_idx[2] - center_idx
                    right_idx_out = right_cross_idx[0]

                # select the larger extension (or whichever exists)
                candidates = []
                if left_radius is not None and left_radius >= 0:
                    candidates.append(('left', left_radius, left_idx_out))
                if right_radius is not None and right_radius >= 0:
                    candidates.append(('right', right_radius, right_idx_out))
                if not candidates:
                    return None, None, None
                # pick maximum extension
                side, radius_val, idx_val = max(candidates, key=lambda x: x[1])
                return float(radius_val), int(idx_val), side

            # Use the fitted center index for consistency
            center_idx_est = int(round(xCenter_fit))
            radius_selected, idx_at_radius, which_side = get_radius_from_threshold(distances, yEst_exp, center_idx=center_idx_est, threshold=1010)

            # If no threshold crossing found, fall back gracefully to half-decay (but capped)
            if radius_selected is None:
                half_decay = math.log(2) / max(lamb, 1e-6)
                radius_selected = float(min(half_decay, 0.5 * len(distances)))
                # pick nearest integer index on the side of fit center (use center+radius rounded)
                idx_at_radius = int(min(len(distances)-1, max(0, center_idx_est + int(round(radius_selected)))))
                which_side = 'right' if (center_idx_est + int(round(radius_selected)) < len(distances)) else 'left'

            # Intensity: pressure at the cutoff radius (from fitted curve)
            # idx_at_radius is the integer index we returned (nearest grid cell). Use that for intensity.
            if idx_at_radius is not None:
                intensity_selected = float(yEst_exp[int(idx_at_radius)])
            else:
                intensity_selected = float(np.min(yEst_exp))

            # --- now continue with assignment/track building as before ---
            assigned = False
            for sid, track in storm_tracks.items():
                last_time, last_lat, last_lon, *_ = track[-1]
                time_diff = (np.datetime64(time.values) - np.datetime64(last_time)).astype('timedelta64[h]').astype(int)
                d_storm = np.sqrt((center_lat - last_lat)**2 + (center_lon - last_lon)**2)
                if time_diff == 1 and d_storm < 1:
                    storm_tracks[sid].append((time.values, center_lat, center_lon, intensity_selected, P_center, radius_selected))
                    assigned = True
                    break
            if not assigned:
                storm_id += 1
                storm_tracks[storm_id].append((time.values, center_lat, center_lon, intensity_selected, P_center, radius_selected))

        

    # Filter and renumber storms for current year
    storm_tracks = {sid: track for sid, track in storm_tracks.items() if len(track) > 7}
    renumbered_storm_tracks = {}
    for new_sid, (old_sid, track) in enumerate(storm_tracks.items(), start=1):
        renumbered_storm_tracks[new_sid] = track
        
    def track_land_count(track):
        """
        track: list of tuples like (time, center_lat, center_lon, P_center, radius_selected)
        returns number of centers classified as 'Land'
        """
        land_count = 0
        for entry in track:
            _, lat, lon, *_ = entry
            if classify_point(lon, lat, land_sea_mask) == "Land":
                land_count += 1
        return land_count

   
    # Store final storm data
    for new_sid, track in renumbered_storm_tracks.items():
        for entry in track:
            time, center_lat, center_lon,intensity_selected,P_center,radius_selected = entry
            id_year = year * 1000 + new_sid
            lsm = classify_point(center_lon, center_lat, land_sea_mask)
            radius_selected1=radius_selected*0.25
            radius_km = radius_selected1 * 111.32 * np.cos(np.radians(center_lat))
            intensity_selected = intensity_selected-P_center
            storm_data.append([id_year, new_sid, time, center_lat, center_lon,intensity_selected, P_center,radius_km,radius_selected1])

df = pd.DataFrame(storm_data,columns=["id_year", "storm_id", "time", "lat", "lon","intensity","central_pressure", "radius","unit_radius"])
def compute_storm_metrics(track_df):
    track_df = track_df.sort_values("time")
    coords = list(zip(track_df["lat"], track_df["lon"]))

    total_length = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(coords[:-1], coords[1:]):
        total_length += geodesic((lat1, lon1), (lat2, lon2)).kilometers

    duration_h = (track_df["time"].iloc[-1] - track_df["time"].iloc[0]) / np.timedelta64(1, "h")
    duration_s = duration_h * 3600
    speed_ms = (total_length * 1000) / duration_s if duration_s > 0 else np.nan
  

    return pd.Series({
        "storm_length_km": total_length,
        "storm_duration_h": duration_h,
        "storm_speed_ms": speed_ms})

storm_metrics = df.groupby("id_year").apply(compute_storm_metrics).reset_index()
df = df.merge(storm_metrics, on="id_year", how="left")
#nearby_ids= df[(df["Distance"] > 0) & (df["Distance"] <= 350)]["id_year"].unique()
#df_near = df[df["id_year"].isin(nearby_ids)] 
# Keep only storms with total path length > 500 km
valid_storms = storm_metrics[storm_metrics["storm_length_km"] > 500]["id_year"].unique()

df = df[df["id_year"].isin(valid_storms)].copy()

# Define bounding box
lon_min, lon_max = -80, -70
lat_min, lat_max = 34, 42

# Select storms with at least one center inside the box
df_box = df[(df["lon"] >= lon_min) & (df["lon"] <= lon_max) &
            (df["lat"] >= lat_min) & (df["lat"] <= lat_max)]
# Get unique storm IDs (id_year) that passed through the box
selected_ids = df_box["id_year"].unique()
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

# Apply to storms near the box
storm_dirs = df_select.groupby("id_year").apply(storm_direction_classification).reset_index()

northward_ids = storm_dirs[
    (storm_dirs["start_lat"] > 38) |
    ((storm_dirs["end_lat"] > storm_dirs["start_lat"]) &
     (storm_dirs["compass"].isin(["N", "NE", "NW"])))]["id_year"].unique()

# Final filtered storms
df_near = df_select[df_select["id_year"].isin(northward_ids)].copy()
print(f"Number of northward-moving storms: {len(df_near['id_year'].unique())}")


df_near["year"] = df_near["id_year"] // 1000
storm_first_time = df_near.groupby(["year", "storm_id"])["time"].min().reset_index() # Find the first time each storm appears
storm_first_time = storm_first_time.sort_values(["year", "time"]) # Sort storms within each year by first appearance
storm_first_time["new_id"] = storm_first_time.groupby("year").cumcount() + 1 # Assign new storm_id starting from 1 within each year

df_near = df_near.merge(storm_first_time[["year", "storm_id", "new_id"]],
                        on=["year", "storm_id"],
                        how="left")

df_near["storm_id"] = df_near["new_id"] # Replace storm_id with new_id

# Recalculate id_year
df_near["id_year"] = df_near["year"] * 1000 + df_near["storm_id"]

# Drop helper columns
df_near = df_near.drop(columns=["year", "new_id"])



output_dir = "/home/biplab/"

df_near.to_parquet(f"{output_dir}coastal_diff_storm_data_{start_year}-{end_year}_lamda.parquet", index=False)



