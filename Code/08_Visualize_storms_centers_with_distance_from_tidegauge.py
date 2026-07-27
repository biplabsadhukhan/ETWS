"""
===============================================================================
Visualization of ETWS Storm Approach Relative to Coastal Orientation
===============================================================================

Description:
    This script visualizes the spatial distribution of Extratropical Winter Storms
    (ETWS) associated with extreme coastal water level events at
    selected NOAA tide gauge stations. 

    For every gauge, the local coastline orientation is estimated from a
    reference coastline, and the coastline normal is computed to divide the
    surrounding area into four directional quadrants (Q1–Q4). Storm locations
    are classified according to their approach direction relative to the
    coastline normal and displayed on an azimuthal equidistant map centered on
    the tide gauge.

    Storm markers are scaled and colored according to the associated peak Sea
    Level Anomaly, providing a visual representation of how storm position and
    approach direction influence coastal surge intensity.

Features:
    - Loads ETWS storm catalogue
    - Identifies extreme storm events using a sigma threshold
    - Computes coastline normals from a reference coastline
    - Calculates storm bearing relative to each tide gauge
    - Classifies storm approach into four coastal quadrants (Q1–Q4)
    - Generates azimuthal equidistant maps centered on each tide gauge
    - Displays concentric distance rings and compass directions
    - Scales marker size and color according to peak Sea Level Anomaly (SLA)
    - Produces publication-quality figures for storm trajectory analysis

Requirements:
    - Python 3.x
    - pandas
    - numpy
    - matplotlib
    - cartopy
    - shapely
    - geopy

Input:
    - ETWS storm catalogue with storm-track information
    - Storm–tide gauge matching dataset containing:
        - Sea Level Anomaly (SLA)
        - Sigma values
        - Storm distance
        - Storm direction
        - Wind stress
        - Storm intensity
        - Storm radius

Output:
    - Publication-quality maps showing:
        - Tide gauge location
        - Storm locations
        - Coastal quadrants (Q1–Q4)
        - Distance rings
        - Compass directions
        - Peak SLA represented by marker size and color

Author:
    Dr. Biplab Sadhukhan
    August 28, 2025
    https://github.com/biplabsadhukhan

Notes:
    - Update the input data path before running the script.
    - Replace placeholder paths (e.g., "/path/to/input/") with directories on
      your local system.
    - The sigma threshold (default: 3σ) used to identify extreme storm events
      can be modified as required.
    - Coastline normals are computed from the supplied reference coastline and
      are used to classify storm approach directions into four coastal
      quadrants (Q1–Q4).
    - Marker size and color represent the peak Sea Level Anomaly (SLA)
      associated with each storm event.
===============================================================================
"""
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.geodesic import Geodesic
from shapely.geometry import Polygon
import matplotlib as mpl

###############################################################################

def compute_bearing(lat1, lon1, lat2, lon2):
  
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlon_rad = np.radians(lon2 - lon1)

    x = np.sin(dlon_rad) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon_rad)

    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360  # Normalize to [0, 360)

def assign_quadrant(storm_angle, normal_angle):
    """
    storm_angle : angle from tide gauge to storm center (0–360)
    normal_angle: coastal normal (0–360)
    """

    # Shift angle so normal becomes 0°
    shifted = (storm_angle - normal_angle) % 360

    if 0 <= shifted < 90:
        return "Q1"
    elif 90 <= shifted < 180:
        return "Q2"
    elif 180 <= shifted < 270:
        return "Q3"
    else:
        return "Q4"

########################################################################################
# ---------------- Paths ----------------
input_file = "/ExtraTropicalWinterStorms/Data/stress_distance_all_tidegauges_1980-2024.parquet"

df = pd.read_parquet(input_file)
###############################################################################
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

############################################
coast_lon = [-80.28, -80.55, -80.54, -80.88, -81.25, -81.43, -81.13, -80.49, -80.01, -79.74, 
             -79.37, -79.21, -78.71, -77.97, -77.73, -77.26, -76.53, -76.06, -75.52, -75.45,
             -75.79, -75.98, -75.89, -75.34, -75.05, -75.07, -74.86, -74.37, -74.11, -73.97, 
             -73.73, -72.85, -71.87, -70.66, -69.93, -70.06, -70.71, -70.60,
             -70.62, -69.57, -68.81, -67.91, -67.17, -66.12, -65.62, -64.84, -64.27, -64.04,
             -63.63, -62.82, -62.54, -61.98, -61.36, -60.97, -60.73, -60.41, -60.08, -59.81]


coast_lat = [27.48, 28.08, 28.48, 29.04, 29.82, 30.85, 31.62, 32.32, 32.61, 32.80, 
             33.02, 33.18, 33.80, 33.85, 34.28, 34.59, 34.60, 35.04, 35.25, 35.63, 
             36.33, 36.91, 37.12, 37.88, 38.40, 38.77, 38.96, 39.39, 39.73, 40.37, 
             40.58, 40.73, 41.06, 41.52, 41.68, 42.07, 42.18, 42.64,
             43.14, 43.81, 44.02, 44.41, 44.68, 43.75, 43.41, 43.83, 44.26, 44.49,
             44.44, 44.72, 44.80, 44.98, 45.16, 45.27, 45.56, 45.63, 45.79, 45.93]
ref_lat=[31.62]
coast_lon = np.array(coast_lon)
coast_lat = np.array(coast_lat)
valid_indices = np.where(np.array(coast_lat) >= ref_lat)[0]
coast_lon1 = coast_lon[valid_indices]
coast_lat1 = coast_lat[valid_indices]

# Compute arc length l from gauge
l = [0.0]
for i in range(1, len(coast_lon1)):
    dist = geodesic((coast_lat1[i-1], coast_lon1[i-1]), (coast_lat1[i], coast_lon1[i])).km
    l.append(l[-1] + dist)
l = np.array(l)
# Reverse the coast path from end → tide gauge
#coast_lat1_rev = coast_lat1[::-1]
#coast_lon1_rev = coast_lon1[::-1]
coast_lat1_rev = coast_lat1
coast_lon1_rev = coast_lon1

coast_bearings = []
for i in range(len(coast_lat1_rev) - 1):
    lat1, lon1 = coast_lat1_rev[i], coast_lon1_rev[i]
    lat2, lon2 = coast_lat1_rev[i + 1], coast_lon1_rev[i + 1]
    bearing = compute_bearing(lat1, lon1, lat2, lon2)
    coast_bearings.append(bearing)
coast_bearings.append(coast_bearings[-1])  # Repeat last to match array length
#coast_bearing_dir = coast_bearings[::-1]
##################################################################################
plt.close()
# ================= YEAR RANGE ====================

start_year, end_year = 1980, 2024
df['time'] = pd.to_datetime(df['time'].values)
df_year = df[(df['time'].dt.year >= start_year) & (df['time'].dt.year <= end_year)]


print(f"Filtering storms from {start_year} to {end_year}")

# Columns that are not SLA/sigma (storm metadata)
storm_meta_cols = [col for col in df.columns if not col.startswith("sla_") and not col.startswith("sigma_")]
storm_tracks_by_station = {}
storm_counts = {}

# ---------------- Extract storms ≥ +1σ ----------------
max_sla_by_station = {}
station_normals = {}
for station in stations:

    
    sigma_col = f"sigma_{station}"
    sla_col = f"sla_{station}"
    dist_col = f"distance_{station}"
    direct_col = f"direction_{station}"
    compass_col = f"compass_{station}"
    scaled_col = f"sla_scaled_{station}"
    
    extreme_storm_ids = df.loc[df[sigma_col] >= 3, "id_year"].unique() # Presently used 3 sigma

    # Extract full tracks for those storms
    station_tracks = df[df["id_year"].isin(extreme_storm_ids)].copy()
    station_tracks = station_tracks[storm_meta_cols + [sla_col, sigma_col,scaled_col]]
    storm_tracks_by_station[station] = station_tracks
    storm_counts[station] = len(extreme_storm_ids)

    df_station = storm_tracks_by_station[station].copy()
    # Index of max sigma per storm
    idx = df_station.groupby("id_year")[sigma_col].idxmax()
    # Select the max rows, including distance + direction + compass
    df_max = df_station.loc[idx, [
       "id_year", "time", sla_col, sigma_col, dist_col, direct_col, compass_col,
       "stress", "storm_speed_ms", "radius", "intensity", scaled_col, "lat", "lon","central_pressure"]]
    
    df_max = df_max.rename(columns={sigma_col: "max_sigma",scaled_col: "sla_scaled"})
    df_max = df_max[df_max["max_sigma"] > 0]
    max_sla_by_station[station] = df_max.reset_index(drop=True)
    
    tg_lat, tg_lon = stations[station]

    # ======= compute tangent line midpoints and normal =======
    mid_lats = (coast_lat1_rev[:-1] + coast_lat1_rev[1:]) / 2
    mid_lons = (coast_lon1_rev[:-1] + coast_lon1_rev[1:]) / 2

    mid_dists = [geodesic((tg_lat, tg_lon), (mlat, mlon)).km
                 for mlat, mlon in zip(mid_lats, mid_lons)]
    mid_idx = np.argmin(mid_dists)

    latA, lonA = coast_lat1_rev[mid_idx],     coast_lon1_rev[mid_idx]
    latB, lonB = coast_lat1_rev[mid_idx + 1], coast_lon1_rev[mid_idx + 1]

    tangent_bearing = compute_bearing(latA, lonA, latB, lonB)
    normal_angle = (tangent_bearing + 90) % 360
    station_normals[station] = normal_angle 
    print(f"{station}: normal = {normal_angle:.2f}°")
    # -------- Print quadrant global angle ranges --------
    print(f"\nAngle Ranges for Station: {station}")
    print(f"Normal angle = {normal_angle:.2f}°")

    Q1_min = (normal_angle + 0) % 360
    Q1_max = (normal_angle + 90) % 360

    Q2_min = (normal_angle + 90) % 360
    Q2_max = (normal_angle + 180) % 360

    Q3_min = (normal_angle + 180) % 360
    Q3_max = (normal_angle + 270) % 360

    Q4_min = (normal_angle + 270) % 360
    Q4_max = (normal_angle + 360) % 360

    print(f"  Q1: {Q1_min:.2f}°  to  {Q1_max:.2f}°")
    print(f"  Q2: {Q2_min:.2f}°  to  {Q2_max:.2f}°")
    print(f"  Q3: {Q3_min:.2f}°  to  {Q3_max:.2f}°")
    print(f"  Q4: {Q4_min:.2f}°  to  {Q4_max:.2f}°\n")

    # ---------------- Assign Quadrants to the max-sigma storm positions ----------------

    # compute bearing from tide gauge -> storm center (vectorized)
    df_max = df_max.copy()
    df_max['bearing_deg'] = compute_bearing(tg_lat, tg_lon, df_max['lat'].values, df_max['lon'].values)

    # assign quadrant using your function (works row-wise)
    df_max['quadrant'] = df_max['bearing_deg'].apply(lambda a: assign_quadrant(a, normal_angle))

    # store back
    max_sla_by_station[station] = df_max

    # ---------- quick diagnostics ----------
    counts = df_max['quadrant'].value_counts().reindex(['Q1','Q2','Q3','Q4']).fillna(0).astype(int)
    #print(f"{station} quadrant counts:\n{counts}\n")
    
results_list = []
plt.close()
# ---------------- PLOTTING ----------------
# Colors per quadrant
quad_colors = {
"Q1": "orange",
"Q2": "brown",
"Q3": "lightblue",
"Q4": "lightgreen"}

def make_sector_polygon(lon_center, lat_center, az1, az2, radius_km, npoints=180):
    """
    Return a shapely Polygon approximating the sector from az1->az2 (degrees)
    out to radius_km (kilometers) using geodesic points.
    az1, az2 in degrees (0=N, increasing clockwise). If az2 < az1 assume wrap.
    """
    # handle wrap: create azimuths that move monotically from az1 to az2 (possibly >360)
    if az2 < az1:
        azs = np.linspace(az1, az2 + 360, npoints)
    else:
        azs = np.linspace(az1, az2, npoints)

    geod = Geodesic()
    outer_pts = []
    for az in azs:
        # Geodesic.direct returns an array of (lon, lat) pairs when points is [(lon, lat)]
        # We pass a single point and extract the first (lon, lat)
        out = geod.direct(points=[(lon_center, lat_center)], azimuths=[az], distances=[radius_km * 1000])[0]
        # out is like [lon, lat, ...] depending on cartopy geodesic version; handle both possibilities
        if len(out) >= 2:
            lon_pt, lat_pt = out[0], out[1]
        else:
            # fallback: treat as tuple
            lon_pt, lat_pt = out
        outer_pts.append((lon_pt, lat_pt))

    # ensure polygon valid: center -> outer ring -> center
    poly_pts = [(lon_center, lat_center)] + outer_pts + [(lon_center, lat_center)]
    return Polygon(poly_pts)



for station in stations:
    print(f"\n=== PLOTTING: {station} ===")
    lat, lon = stations[station]

    df_station_all = max_sla_by_station[station].copy()
    

    # Use station dataframe for sizes/colors
    sla_vals = df_station_all[f"sla_{station}"].copy()
    sla_min, sla_max = sla_vals.min(), sla_vals.max()

    def scale_size(x, min_size=5, max_size=500):
        if pd.isna(x):
            return 0
        if sla_max == sla_min:
            return (min_size + max_size) / 2
        return min_size + (x - sla_min) / (sla_max - sla_min) * (max_size - min_size)

    sizes = sla_vals.apply(scale_size).values
    lats = df_station_all["lat"].values
    lons = df_station_all["lon"].values

    fig = plt.figure(figsize=(25, 12))
    
    ax1 = fig.add_axes([0.00001, 0.1, 0.6, 0.8], projection=ccrs.AzimuthalEquidistant(central_latitude=lat,
                                                                                   central_longitude=lon))
    ax1.set_title(f"{station} – Storm Locations by Quadrant", fontsize=20, fontweight='bold')

    # Base map features
    ax1.add_feature(cfeature.LAND, facecolor="lightgray")
    ax1.add_feature(cfeature.COASTLINE, linewidth=1)
    ax1.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
    ax1.patch.set_edgecolor('black')
    ax1.patch.set_linewidth(2)

    # extent (same as your original)
    max_radius_km = 1800
    pad_km = 200
    extent_deg = (max_radius_km + pad_km) / 111
    ax1.set_extent([lon - extent_deg, lon + extent_deg, lat - extent_deg, lat + extent_deg], crs=ccrs.PlateCarree())
    ax1.set_aspect("equal", adjustable="box")
    null=4
    #iso200 = ax1.contour(bath_lon[::null],bath_lat[::null],elev[::null, ::null],levels=[-200],colors="blue",linewidths=1.6,linestyles="--",transform=ccrs.PlateCarree())
 
    normal_angle = station_normals[station]

    # Quadrant boundaries (az1, az2)
    Q1 = (normal_angle + 0)   % 360, (normal_angle + 90)  % 360
    Q2 = (normal_angle + 90)  % 360, (normal_angle + 180) % 360
    Q3 = (normal_angle + 180) % 360, (normal_angle + 270) % 360
    Q4 = (normal_angle + 270) % 360, (normal_angle + 360) % 360

    quad_bounds = {
        "Q1": Q1,
        "Q2": Q2,
        "Q3": Q3,
        "Q4": Q4
    }

    # Draw each quadrant polygon
    for qname, (az1, az2) in quad_bounds.items():

        poly = make_sector_polygon(lon, lat, az1, az2, radius_km=max_radius_km, npoints=200)
        ax1.add_geometries([poly], crs=ccrs.PlateCarree(),
                           facecolor=quad_colors.get(qname, "none"), alpha=0.20, edgecolor=None, zorder=10)
        # midpoint bearing for label (works with wrap)
        width = (az2 - az1) % 360
        label_bearing = (az1 + width / 2) % 360
        label_point = Geodesic().direct(points=[(lon, lat)], azimuths=[label_bearing], distances=[(max_radius_km-200) * 1000])[0]
        # label_point returns [lon, lat, ...]; extract first two
        lx, ly = label_point[0], label_point[1]
        ax1.text(lx, ly, qname, fontsize=18, fontweight="bold", transform=ccrs.PlateCarree(), ha="center", va="center", color="red", zorder=50)


    ax1.plot(lon, lat, "r^", markersize=18, transform=ccrs.PlateCarree(), label="Station", zorder=350)
        
    gd = Geodesic()
    compass_labels = {0: "N", 45: "NE", 90: "E", 135: "SE",
        180: "S", 225: "SW", 270: "W", 315: "NW"}
    radii_km = np.arange(200, 2000, 200)
    for r in radii_km:
        circle = gd.circle(lon=lon, lat=lat, radius=r * 1000, n_samples=180)
        poly = Polygon(circle)
        ax1.plot(*poly.exterior.xy, color="black", linestyle="--", linewidth=0.7,
                transform=ccrs.Geodetic())
        # --- Add ticks every 45° ---
        for bearing in compass_labels.keys():
            tick = gd.direct(points=[(lon, lat)], azimuths=[bearing], distances=[r * 1000])
            tick_lon, tick_lat = tick[0][0], tick[0][1]

            # A slightly shorter line for the tick mark
            tick_inner = gd.direct(points=[(lon, lat)], azimuths=[bearing], distances=[(r - 20) * 1000])
            inner_lon, inner_lat = tick_inner[0][0], tick_inner[0][1]

            ax1.plot([inner_lon, tick_lon], [inner_lat, tick_lat],
                    color="black", linewidth=1.8, transform=ccrs.Geodetic()) # line for compass
   
        label_pos = gd.direct(points=[(lon, lat)], azimuths=[315], distances=[(r + 40) * 1000])
        label_lon, label_lat = label_pos[0][0], label_pos[0][1]
        
        ax1.text(label_lon, label_lat, f"{r} ",fontsize=12,
                ha="right", va="top",  # anchors text along NW direction
                transform=ccrs.PlateCarree())
        cross_directions = [0, 45, 90, 135, 180, 225, 270, 315]  # bearings
        outer_r = radii_km[-1]  # outermost ring (e.g., 1800 km)

        for bearing in cross_directions:
            # End point at the outer radius
            end = gd.direct(points=[(lon, lat)], azimuths=[bearing], distances=[outer_r * 1000])
            end_lon, end_lat = end[0][0], end[0][1]

            # Draw line from station → endpoint
            # ax1.plot([lon, end_lon], [lat, end_lat],linestyle=":", color="red", linewidth=1,
            #         transform=ccrs.Geodetic())   # line for comapss
    # --- Compass labels just beyond the last circle ---
    outer_r = radii_km[-1] + 100  # little outside the largest ring
    for bearing, label in compass_labels.items():
        pos = gd.direct(points=[(lon, lat)], azimuths=[bearing], distances=[outer_r * 1000])
        pos_lon, pos_lat = pos[0][0], pos[0][1]
        ax1.text(pos_lon, pos_lat, label, fontsize=14, fontweight="bold",
               ha="center", va="center", transform=ccrs.PlateCarree())
   
    # --- Scatter storms using station dataframe ---
    # Filter only finite scaled values
    good_mask = np.isfinite(sla_vals.values)
    order = np.argsort(sizes[good_mask])
    if good_mask.sum() == 0:
        print(f"No finite sla_scaled values for {station}.")
    else:
        cmap = plt.get_cmap("summer_r")
        norm = mpl.colors.Normalize(vmin=0.5, vmax=1.5)
        #norm = mpl.colors.Normalize(vmin=sla_vals.min(), vmax=sla_vals.max())
        # Sort points by size (small first, large last)

        sc = ax1.scatter(lons[good_mask][order],lats[good_mask][order],c=sla_vals.values[good_mask][order],
            cmap=cmap, norm=norm,s=sizes[good_mask][order],
            alpha=0.85,edgecolor="k",
            linewidth=1,
            zorder=300,transform=ccrs.PlateCarree())

    plt.show()
    
    # output_dir = "/home/biplab/Documents/Work_1/Research/Figure_3/t3/regression/"
    # os.makedirs(output_dir, exist_ok=True)
    # save_path = os.path.join(output_dir, f"{station}_quadrant_sla3_highpass.png")
    # plt.savefig(save_path, dpi=300,bbox_inches="tight")



#