# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # OTPW Distinct Feature Values
# MAGIC This notebook is retreiving the distinct values present in features of the full OTPW dataset so that external data source exploration can be reduced to just the feature values that are relevant *(i.e. only looking at weather stations that are relevant to airports)*

# COMMAND ----------

# importing custom functions
import funcs

from pyspark.sql.functions import col, expr, from_json
from pyspark.sql.types import TimestampType

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 1) Quantifying Weather Stations
# MAGIC Here we are quantifying all the unique weather stations that were used in the OTPW joined data.  
# MAGIC This list will help us in filtering down the raw weather data to just the relevant weather stations.
# MAGIC
# MAGIC **This was performed on the entire dataset so that all weather stations are accounted for.**

# COMMAND ----------

# the 261 course blob storage is mounted here.
mids261_mount_path      = "/mnt/mids-w261"
display(dbutils.fs.ls(f"{mids261_mount_path}/"))


# COMMAND ----------

# Reading the entire OTPW dataset
otpw = spark.read.format("csv").option("header","true").load(f"{mids261_mount_path}/OTPW_60M/")

# Quantifying the stations mapped. 
origin_st = ['origin_station_id', 'origin_station_name', 'origin_region']
dest_st =  ['dest_station_id', 'dest_station_name', 'dest_region']

# capturing distinct origin weather stations, renaming columns
origins = otpw[origin_st].distinct()
origins = origins.withColumnRenamed(origin_st[0], 'STATION').withColumnRenamed(origin_st[1], 'NAME').withColumnRenamed(origin_st[2], 'REGION')

# capturing distinct destination weather stations, renaming columns
dests = otpw[dest_st].distinct()
dests = dests.withColumnRenamed(dest_st[0], 'STATION').withColumnRenamed(dest_st[1], 'NAME').withColumnRenamed(origin_st[2], 'REGION')

# combining origin and destination weather stations to one dataset, and reducing to distinct locations
all_stations = origins.union(dests)
all_stations = all_stations.distinct()
print(f"There are {all_stations.count()} unique weather stations used.")

# write the station list in parquet to blob storage
funcs.write_parquet_to_blob(all_stations, 'ES/OTPW/All_Stations')

# COMMAND ----------

# there is one duplicate station because of region misalignment
team_blob_url = funcs.blob_connect()
all_stations = spark.read.parquet(f"{team_blob_url}/ES/OTPW/All_Stations")
repeated_stations = all_stations.groupBy("STATION").count().orderBy('count', ascending=False).show()

# COMMAND ----------

# reviewing station with two regions
all_stations[all_stations['STATION'] == '91197521510'].show()

# COMMAND ----------

# removing one region from specific station
all_stations = all_stations[(all_stations['STATION'] != '91197521510') | (all_stations['REGION'] != 'AS-U-A')]

# rewriting to blob storage
funcs.write_parquet_to_blob(all_stations, 'ES/OTPW/All_Stations')

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 2) Quantifying All Airlines
# MAGIC Here we are quantifying all the unique airlines that are represented in the entire dataset. This will be used to map airline stock prices to see if that has any correlation to performance. 
# MAGIC
# MAGIC **This was performed on the entire dataset so that all weather stations are accounted for.**

# COMMAND ----------

# Reading the entire OTPW dataset
otpw = spark.read.format("csv").option("header","true").load(f"{mids261_mount_path}/OTPW_3M/")

# airlines features
airlines = ['OP_UNIQUE_CARRIER','OP_CARRIER_AIRLINE_ID']

# capturing distinct airlines
unique_airlines = otpw[airlines].distinct()

print(f"There are {unique_airlines.count()} unique airlines used.")

display(unique_airlines)
# # write the station list in parquet to blob storage
# funcs.write_parquet_to_blob(all_stations, 'ES/OTPW/All_Airlines')

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 2) Exploring UTC to Local Conversion
# MAGIC The OTPW dataset has both local and universal timestamps.  The weather data only has local timestamps but we need universal to properly join both the departure and arrival weather.  Here we are looking to reverse engineer the difference between the two timestamps in the OTPW to apply to the weather data. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Capturing Conversion for Weather Stations

# COMMAND ----------

# Reading the 12 Month OTPW dataset
# It was discovered that the larger datasets do not include the local departure time field
otpw = spark.read.format("csv").option("header","true").load(f"{mids261_mount_path}/OTPW_12M/")

# Quantifying the stations mapped. 
stations = ['origin_station_id','origin_station_name', 'origin_region', 'sched_depart_date_time',
 'sched_depart_date_time_UTC']

# reducing dataset to relevant columns
otpw = otpw[stations]

# casting timestamp columns to timestamp
otpw = otpw.withColumn('sched_depart_date_time',col('sched_depart_date_time').cast("timestamp")).withColumn('sched_depart_date_time_UTC',col('sched_depart_date_time_UTC').cast("timestamp"))

# adding a time offset column between local and UTC
otpw = otpw.withColumn("time_offset", expr("sched_depart_date_time - sched_depart_date_time_UTC"))

# Extract the "seconds" field
otpw = otpw.withColumn("seconds_offset", col("time_offset").cast('int'))

# reducing the dataframe to just the distinct offsets
otpw = otpw.select('origin_station_id','origin_station_name', 'origin_region', 'seconds_offset').distinct()

# # renaming columns
otpw = otpw.withColumnRenamed('origin_station_id', 'STATION').withColumnRenamed('origin_station_name', 'NAME')

# US local times lag UTC, and daylight savings brings US one hour closer to UTC.
# so the larger offsets by station represent the local times ignoring DST. 
# reducing the df to just be standardized local time. 
otpw = otpw.groupBy('STATION', 'NAME', 'origin_region').min('seconds_offset')

print(f"There are {otpw.count()} unique weather stations and time_offsets used.")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Imputing Conversion for Weather Stations Not Accounted for

# COMMAND ----------

# Reading the all stations dataset
team_blob_url = funcs.blob_connect()
all_stations = spark.read.parquet(f"{team_blob_url}/ES/OTPW/All_Stations")

# checking the distinct regions present in the entire dataset
regions = all_stations.select('REGION').distinct()
regions.count()

# COMMAND ----------

# Reading the timestamp weather stations
stamped_stations = otpw

# Capturing all the distinct regions that are timestamped
stamped_regions = stamped_stations.select('origin_region', 'min(seconds_offset)').distinct()

# reducing the df to just be smallest offset, which represents standard time
stamped_regions = stamped_regions.groupBy('origin_region').min('min(seconds_offset)')
display(stamped_regions)

# COMMAND ----------

# creating a list of all the stations that already have a timestamp
station_list = stamped_stations.rdd.map(lambda x: x[0]).collect()

# quantifying the stations that still need a timestamp
stations_left = all_stations[~all_stations['STATION'].isin(station_list)]

# using the regional timestamp for the remaining stations
# by joining to regions general timestamp
stations_left = stations_left.join(stamped_regions,
                                   stations_left.REGION == stamped_regions.origin_region,
                                   how = 'left')

# dropping extra columns
stations_left = stations_left.drop('REGION')

# renaming columns
stations_left = stations_left.withColumnRenamed('min(min(seconds_offset))', 'min(seconds_offset)')

# combining all stations together
all_station_timestamps = stamped_stations.union(stations_left)

# renaming columns
all_station_timestamps = all_station_timestamps.withColumnRenamed('origin_region', 'REGION') \
                                                .withColumnRenamed('min(seconds_offset)', 'UTC_offset')

# # write the station list in parquet to blob storage
funcs.write_parquet_to_blob(all_station_timestamps, 'ES/OTPW/Timestamps')

# COMMAND ----------

