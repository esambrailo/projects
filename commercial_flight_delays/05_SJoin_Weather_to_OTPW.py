# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Joining Weather Features to OTPW Dataset

# COMMAND ----------

# importing custom functions
import funcs
from pyspark.sql import functions as f
from pyspark.sql.window import Window
from pyspark.sql.types import  TimestampType

# the 261 course blob storage is mounted here.
mids261_mount_path      = "/mnt/mids-w261"
display(dbutils.fs.ls(f"{mids261_mount_path}/"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 1) Dropping Existing Weather Features

# COMMAND ----------

# Reading the OTPW dataset
otpw = spark.read.format("csv").option("header","true").load(f"{mids261_mount_path}/OTPW_60M/")
# dropping the existing weather observation columns
otpw = funcs.drop_existing_weather(otpw)
display(otpw)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 2) Joining Daily Features
# MAGIC
# MAGIC Joining the Daily Summary Weather features to the OTPW dataset.  This join is being done for both the arrival and departure locations using the identified weather stations for both. 
# MAGIC
# MAGIC **For Departure (origin) Location**
# MAGIC
# MAGIC |OPTW|Daily Weather|
# MAGIC |---|---|
# MAGIC |'FL_DATE'|'Next_Date'|
# MAGIC |'origin_station_id'|'STATION'|
# MAGIC
# MAGIC **For Arrival (dest) Location**
# MAGIC
# MAGIC |OPTW|Daily Weather|
# MAGIC |---|---|
# MAGIC |'FL_DATE'|'Next_Date'|
# MAGIC |'dest_station_id'|'STATION'|
# MAGIC

# COMMAND ----------

# printing count of OTPW records
print(f"There are {otpw.count()} records in the OTPW dataset prior to joins.")

# read in daily weather data from parquet
team_blob_url = funcs.blob_connect()
daily_weather = spark.read.parquet(f"{team_blob_url}/ES/Weather/Updated_SOD")

origin_weather = funcs.add_prefix(daily_weather, 'origin_')
# joining departure location data
temp_otpw = otpw.join(origin_weather, 
                         (otpw.FL_DATE == origin_weather.origin_Next_Date) & 
                         (otpw.origin_station_id == origin_weather.origin_STATION), 
                         how = 'left')

dest_weather = funcs.add_prefix(daily_weather, 'dest_')
# joining arrival location data
updated_otpw = temp_otpw.join(dest_weather, 
                         (temp_otpw.FL_DATE == dest_weather.dest_Next_Date) & 
                         (temp_otpw.dest_station_id == dest_weather.dest_STATION), 
                         how = 'left')

# printing count of OTPW records
print(f"There are {updated_otpw.count()} records in the OTPW dataset after the joins.")

# COMMAND ----------

display(updated_otpw)

# COMMAND ----------

# dropping unnecessary columns
updated_otpw = updated_otpw.drop(*('origin_STATION',
 'origin_DATE',
 'origin_Next_Date',
 'dest_STATION',
 'dest_DATE',
 'dest_Next_Date'))

# writing data as parquet to blob
funcs.write_parquet_to_blob(updated_otpw, 'ES/OTPW/partial_update')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 3) Joining Hourly Features
# MAGIC
# MAGIC Joining the Hourly Observation Weather features to the OTPW dataset.  This join is being done for both the arrival and departure locations using the identified weather stations for both. 
# MAGIC
# MAGIC **For Departure (origin) Location**
# MAGIC
# MAGIC |OPTW|Hourly Weather| Notes|
# MAGIC |---|---|---|
# MAGIC |'four_hours_prior_depart_UTC' < > 'three_hours_prior_depart_UTC'|'UTC'|*Both sides of join are UTC time*|
# MAGIC |'origin_station_id'|'STATION'|
# MAGIC
# MAGIC **For Arrival (dest) Location**
# MAGIC
# MAGIC |OPTW|Hourly Weather| Notes|
# MAGIC |---|---|---|
# MAGIC |'four_hours_prior_depart_UTC' < > 'three_hours_prior_depart_UTC'|'UTC'|*Both sides of join are UTC time*|
# MAGIC |'dest_station_id'|'STATION'|
# MAGIC

# COMMAND ----------

# # read in data from parquet
team_blob_url = funcs.blob_connect()
otpw = spark.read.parquet(f"{team_blob_url}/ES/OTPW/partial_update")

# adding a feature for three hours from departure in UTC
otpw = otpw.withColumn('two_hours_long', f.col('two_hours_prior_depart_UTC').cast(TimestampType()).cast('Long')) \
            .withColumn('three_hours_prior_depart_UTC', f.expr("two_hours_long - 60*60")) \
            .withColumn('three_hours_prior_depart_UTC', f.col('three_hours_prior_depart_UTC').cast(TimestampType())) \
            .drop('two_hours_long')

# printing count of OTPW records
print(f"There are {otpw.count()} records in the total OTPW dataset.")

# read in daily weather data from parquet
team_blob_url = funcs.blob_connect()
hourly_weather = spark.read.parquet(f"{team_blob_url}/ES/Weather/Updated_Hourly")

# printing count of OTPW records
print(f"There are {hourly_weather.count()} records in the total hourly weather dataset.")

# reducing dataset to smaller chunks
# otpw = otpw[otpw['FL_DATE'] < '2016-01-01']
# hourly_weather = hourly_weather[hourly_weather['DATE'] < '2016-01-01']
print('-'*70)
# printing count of OTPW records
print(f"There are {otpw.count()} records in the reduced OTPW dataset prior to joins.")
# printing count of OTPW records
print(f"There are {hourly_weather.count()} records in the reduced hourly weather dataset prior to joins.")


# COMMAND ----------

# joining hourly weather observations for both the origin airports 
temp_otpw = funcs.joining_hourly(otpw, hourly_weather, 'origin')

# re-sorting the dataframe for next step
temp_otpw = temp_otpw.sort("dest_station_id", "three_hours_prior_depart_UTC", ascending=[True, False]) 

# joining hourly weather observations for both the destination airports 
final_otpw = funcs.joining_hourly(temp_otpw, hourly_weather, 'dest')

# removing any residual duplications that occured from join by
# dropping duplicate flight records,
final_otpw = final_otpw.dropDuplicates(['FL_DATE','ORIGIN_AIRPORT_ID', 'OP_CARRIER_FL_NUM', 'TAIL_NUM','sched_depart_date_time_UTC'])

# printing count of OTPW records
print(f"There are {final_otpw.count()} records in the OTPW dataset after the joins.")

def drop_excess_join_cols(df):
    '''Function for dropping the pre-existing weather features from OTPW dataset.'''
    drop_columns = ['origin_STATION',
                    'origin_LATITUDE',
                    'origin_LONGITUDE',
                    'dest_STATION',
                    'dest_LATITUDE',
                    'dest_LONGITUDE']
    reduced_df = df.drop(*drop_columns)
    return reduced_df

final_otpw = drop_excess_join_cols(final_otpw)

# writing data as parquet to blob
funcs.write_parquet_to_blob(final_otpw, 'ES/new_joins/5Y')

# COMMAND ----------

# # read in data from parquet
team_blob_url = funcs.blob_connect()
otpw = spark.read.parquet(f"{team_blob_url}/LH/1yr_clean_temp_2")
otpw.columns

# COMMAND ----------

