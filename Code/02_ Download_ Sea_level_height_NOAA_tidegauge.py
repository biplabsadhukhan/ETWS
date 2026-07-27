"""
===============================================================================
NOAA CO-OPS Hourly Tide Gauge Water Level Download Script
===============================================================================

Description:
    This script downloads hourly water level (sea level anomaly) observations
    from NOAA CO-OPS tide gauge stations along the U.S. East Coast using the
    NOAA CO-OPS Python API (`noaa_coops`).

    The script retrieves hourly water level data referenced to the Mean Lower
    Low Water (MLLW) datum in metric units for the period 1980–2024. Data are
    downloaded for all selected stations from Maine to South Carolina and
    saved individually as compressed Apache Parquet (.parquet.gzip) files.

Features:
    - Downloads hourly water level observations
    - Retrieves station metadata (latitude and longitude)
    - Supports custom station lists or geographic selection
    - Automatically skips stations that have already been downloaded
    - Removes missing values before saving
    - Saves data in compressed Parquet format for efficient storage

Requirements:
    - Python 3.x
    - pandas
    - numpy
    - matplotlib
    - noaa_coops
    - pyarrow (for Parquet output)

Data Source:
    NOAA Center for Operational Oceanographic Products and Services (CO-OPS)
    https://tidesandcurrents.noaa.gov/

Output:
    One compressed Parquet file per station containing:
        - date_time
        - water level (meters)
        - latitude
        - longitude

Author:
    Biplab Sadhukhan
    DEC 24 18:21:48 2024
GitHub:
    https://github.com/biplabsadhukhan

Notes:
    - Modify 'fileNameOut' to specify your local output directory.
    - Internet access is required to query the NOAA CO-OPS API.
    - Some stations may not contain data for the requested time period and
      will be skipped automatically.
===============================================================================
"""
import datetime
from pylab import *
from numpy import *
import os
import pandas as ps

# Import the main CO-OPS API helper class
#from CoopsApi import CoopsApi
from noaa_coops import Station, get_stations_from_bbox

# Data parameters
#if True, define stations manually, if False, choose by bounding box
if False:
    #stationName IS NO LONGER USED, AND IS NOT NEEDED
    data2get=[
        dict(stationName='FortPoint_NH',station_id="8423898"),
        dict(stationName='Providence_RI',station_id="8454000"),
        dict(stationName='FallRiver_MA',station_id="8447386"),
        dict(stationName='TheBattery_NY',station_id="8518750"),
        dict(stationName='NewLondon_CT',station_id='8461490'),
        dict(stationName='Montauk_NY',station_id='8510560'),
        dict(stationName='WoodsHole_MA',station_id='8447930'),
        dict(stationName='SeaveyIsland_ME',station_id='8419870'),
        dict(stationName='Newport_RI',station_id='8452660'),
        dict(stationName='QuonsetPoint_RI',station_id='8454049'),
        dict(stationName='FallRiver_MA',station_id='8447386'),
        dict(stationName='ConimicutLight_RI',station_id='8452944')
    ]
elif False:
    #define a bounding box and get stations
    #this seems to miss many of the available stations,
    #DO NOT USE
    stations = get_stations_from_bbox(lat_coords=[25, 46], lon_coords=[-82,-61]) #Doppio range
    print(stations)
    data2get=[{'station_id':p} for p in stations]
else:
    #sigh, define by hand any from NC to Maine, any with data 1/1/2000 on. Work done on 4/2024
    stations=[8410140,8410834,8411250,8410864,8411060,8413320,8414612,8416921,8415709,8417087, #ME
              8417177,8414249,8418445,8419870,8418150,8419317,8418606,8414888,8415490,8416731, #ME
              8419870,8423898, #NH
              8441241,8443970,8444788,8446121,8447173,8447495,8444162,8447180,8447259,8447368, #MA
              8447416,8443725,8444525,8447191,8447270,8447295,8447435,8447636,8448875,8447685, #MA
              8448558,8449130,8447930,8448248,8447712, #MA
              8452660,8452154,8452944,8459681,8454000,8454049,8454578,8455137,8450768,8453742, #RI
              8460751,8461490,8463701,8465705,8467150,8468191,8467373,8468448,8467726,8469198, #CT
              8511629,8512354,8511671,8512451,8512735,8510560,8511907,8512668,8512987,8515186, #NY
              8516607,8516663,8516891,8513388,8514779,8516402,8516945,8513398,8515786,8516501, #NY
              8516881,8516990,8516155,8517251,8517847,8518639,8518687,8518750,8517756,8518526, #NY
              8518643,8518902,8517201,8518668,8518699,8518934,8518995,8519436,8518962,8519050, #NY
              8530882,8518091,8518924,8510448,8512769,8513825,8519483,8516661, #NY
              8531142,8531680,8531804,8533365,8533615,8533051,8534319,8534720,8535055,8535375, #NJ
              8535581,8536110,8537121,8539094,8535835,8534770,8534836,8535419, #NJ
              8551910,8551762,8554399,8555889,8558814,8557380,8559957, #DE
              8540433,8538512,8545240,8546252,8539993, #PA
              8570283,8570691,8571421,8571858,8571117,8571359,8571559,8571702,8571579,8571773, #MD
              8573704,8572467,8572955,8573349,8574070,8574680,8575512,8572669,8573364,8574683, #MD
              8577330,8577004,8579542,8570280,8630315,8630308,8571892, #MD
              8631044,8632200,8635257,8636653,8636941,8637712,8638339,8632869, #VA
              8635027,8635750,8636580,8637624,8633532,8635150,8635985,8637689,8638017,8638424, #VA
              8638433,8638450,8638481,8638495,8638671,8638445,8638464,8638489,8638610,8638863, #VA
              8638901,8639207,8639348, #VA
              8594900, #DC
              8650249,8651370,8651687,8652226,8652437,8652648,8650946,8651817,8652547,8651097, #NC
              8651986,8652247,8652587,8652905,8653244,8653395,8653795,8654572,8654875,8655353, #NC
              8653084,8653471,8654400,8654769,8655133,8655875,8656467,8653215,8653365,8653717, #NC
              8654467,8654812,8655151,8656483,8656613,8657002,8658120,8656590,8658145,8657813, #NC
              8659084,8659665,8659897,8659414,8651538,8656201,8653951,8658163, #NC
              8661070,8661437,8661419,8661684,8661989,8662071,8663219,8663858,8664941,8665530, #SC
              8664662,8668498 #SC
              ]
    data2get=[{'station_id':p} for p in stations]

#add start and end dates to each entry
for n in range(len(data2get)):
    data2get[n]['start_date']=datetime.datetime(1980,1,1)
    data2get[n]['end_date']=datetime.datetime(2024,12,31)


#get all data points
for theStation in data2get:

    # Create a CoopsApi instance and query the API for data. Check for errors
    #new NOAA_COOPS api
    thePlace=Station(id=theStation['station_id'])
    theLat=thePlace.metadata['lat']
    theLon=thePlace.metadata['lng']

    #make a file name
    stationName=thePlace.name+'_'+thePlace.state
    stationName=stationName.replace(' ','_')
    fileNameOut='Code/new_tidegauge/'+stationName+'.prq.gzip' # Change according to your directory to save file 

    if not os.path.exists(fileNameOut):

        print('\nStarting with',stationName,theStation['station_id'])

        try:
            jnk=thePlace.get_data(
                begin_date=theStation['start_date'].strftime("%Y%m%d"),
                end_date=theStation['end_date'].strftime("%Y%m%d"),
                product="hourly_height",
                datum="MLW",
                units="metric",
                time_zone="gmt")

            #make a data frame in format I want
            #it will probably bite me in the ass that I don't save or check the quality flags....
            output=ps.DataFrame({'date_time':jnk.index,'value':jnk['v'],'lat':theLat,'lon':theLon})
            gotData=True
        except:
            print('Failed to find data in time range for',stationName)
            gotData=False
            #assert False,'well, what data is there?'

        #assert False,'what data is there?'

        if gotData:
            # Return keys of dictionary for specific product and request
            print("Returned keys:" + str(output.keys()))


            # Let's iterate through the timestamps and water level values
            #num_values=len(output['date_time'])
            #i=0
            #while i<num_values:
            #    print(str(output['date_time'][i]),output['value'][i])
            #    i=i+1

            indx=isfinite(output['value'].values)
            output=output.loc[indx]

            print('First and last data of',stationName)
            print('   ',str(output['date_time'][0]),output['value'][0])
            print('   ',str(output['date_time'].iloc[-1]),output['value'].iloc[-1])
            print('   ',len(output['date_time'])/24,'days of data')


            #save data
            output=output.reset_index()
            output=output.drop('t',axis=1)
            output.to_parquet(fileNameOut,compression='gzip')

            #plot
            #figure()
           # clf()
            #style.use('ggplot')
            #plot(output.date_time,output.value)
            #title(stationName)
            #draw()
            #show(block=False)
            #pause(0.1)

            print('done with',stationName)
            #print(' ')

