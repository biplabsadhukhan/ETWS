# ETWS tracks and correspondind surge
A Python-based workflow for identifying, tracking, and analyzing Extratropical Winter Storms (ETWS) using ERA5 reanalysis and NOAA tide gauge observations. This repository provides tools to build storm catalogues, process tide gauge records, compute storm characteristics, quantify storm surge responses, and generate publication-quality figures for coastal hazard research.

Overview

This repository contains scripts that
01	Download ERA5 Mean Sea Level Pressure (MSLP) reanalysis data from the Copernicus  Climate Data Store (CDS).
02	Download hourly NOAA CO-OPS tide gauge observations.
03	Remove astronomical tides using UTide harmonic analysis to obtain Sea Level Anomaly (SLA). Apply cosine-window high-pass filtering to isolate storm-related sea level signals.
04	Detect and track Extratropical Winter Storms (ETWS) from ERA5 MSLP fields. Compute storm characteristics including:
•	Central pressure
•	Storm radius
•	Pressure deficit (intensity)
•	Storm duration
•	Translation speed
•	Track length
•	Storm direction

04_1 Estimate storm radius and intensity using both threshold-based and double-exponential pressure profile methods.
05	Match storm tracks with coastal tide gauge observations.
06	Estimate geostrophic wind stress. Calculate storm-to-gauge distance and approach direction.
07	Compute storm surge duration.
08	Classify storm approach relative to local coastline orientation. Produce publication-quality maps and visualizations.
09	Scripts contains code for regression analysis and visualization of the scatter and coefficient of regression.
Note: The numbers in front of descriptions are script numbers mentioned in ExtraTropicalWinterStorms/code/ 

Data Sources
ERA5 Reanalysis
•	Variable: Mean Sea Level Pressure (MSLP)
•	Temporal resolution: Hourly
•	Source: Copernicus Climate Data Store (CDS)
NOAA CO-OPS Tide Gauges
•	Hourly water level observations
•	Datum: Mean Lower Low Water (MLLW/MLW)
•	Units: meters

This work is supported by NSF project Award No - 2316271. 

Python software
All Python scripts required for processing, analyzing, and visualizing the data are located in the code/ directory. The scripts are extensively commented to make the workflow easy to understand and modify for your own applications.
Before running the scripts, install the required Python packages listed in requirements.txt (or install the necessary dependencies manually using pip). Most scripts rely on common scientific Python libraries, including:

numpy
pandas
matplotlib
scipy
xarray
cartopy
geopy
shapely
statsmodels
rasterio
utide
pyarrow
Detailed documentation, along with the well-documented example scripts and utility functions included in this repository, should provide everything needed to get started. If you encounter any issues or have questions about the code or workflow, please feel free to open a GitHub Issue or Email me.

Suggested Citation
If you use this repository in your research, please cite it as:

Sadhukhan, B. (2026). ExtraTropicalWinterStorms: Python tools for analyzing extratropical cyclone tracks, storm surge, and tide gauge observations (Version 1.0) [Computer software]. GitHub. https://github.com/biplabsadhukhan/ExtraTropicalWinterStorms

If this software contributes to your research, please consider citing both this repository and any associated journal publications describing the methodology.

Suggested Acknowledgment


Please direct questions to Dr. Biplab Sadhukhan, University of New Hampshire, :  biplabsadhukhan3@gmail.com.
