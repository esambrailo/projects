# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Hourly Weather Observations Dataset EDA
# MAGIC This notebook is to explore the dataset we created for the hourly weather observations from the original raw dataset. 
# MAGIC

# COMMAND ----------

# importing modules for smaller summary datasets
import pandas as pd
import numpy as np
# importing visualization modules
import seaborn as sns
import matplotlib.pyplot as plt
# importing custom functions
import funcs

from pyspark.sql.functions import col, udf, expr
from pyspark.sql import functions as f
from pyspark.ml.stat import Correlation
from pyspark.sql.types import DateType, TimestampType, StringType
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 1) Read-in and address nulls/types

# COMMAND ----------

# # read in data from parquet
team_blob_url = funcs.blob_connect()

hourly_weather = spark.read.parquet(f"{team_blob_url}/ES/Weather/Hourly")

# grouping features
station_attr = ['STATION', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'NAME', 'SOURCE', 'REPORT_TYPE']
hourly_cats = ['HourlyPresentWeatherType', 'HourlySkyConditions']
hourly_obs = [feature for feature in hourly_weather.columns if "Hourly" in feature]
for feature in hourly_cats:
    hourly_obs.remove(feature)

# reducing dataframe to relevant columns
hourly_weather = hourly_weather[['DATE'] + station_attr + hourly_cats + hourly_obs]

# removing trace values
hourly_weather = hourly_weather.replace('T', '0.005')

# casting daily features as float
for col_name in hourly_obs:
    hourly_weather = hourly_weather.withColumn(col_name, col(col_name).cast('float'))

# checking values for each feature
describe = funcs.describe_table(hourly_weather)
display(describe)

# COMMAND ----------

hourly_weather.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2) Reviewing Report Types

# COMMAND ----------

display(hourly_weather.select('REPORT_TYPE').distinct())

# SHEF – Standard Hydrologic Exchange Format 
# FM-12 = SYNOP Report of surface observation form a fixed land station 
# FM-15 = METAR Aviation routine weather report 
# FM-16 = SPECI Aviation selected special weather report 
# SY-MT = Synoptic and METAR merged report 

# COMMAND ----------

feature ='REPORT_TYPE'
funcs.histogram(hourly_weather, feature)

# COMMAND ----------

# looking at records with a FM-16 report type
display(hourly_weather[hourly_weather['REPORT_TYPE']== 'FM-16'])
# looking at records with a FM-12 report type
display(hourly_weather[hourly_weather['REPORT_TYPE']== 'FM-12'])

print('='*60)
print('Counts of Unique Stations:')
print('Total Unique Airports: ', hourly_weather[['STATION','NAME']].distinct().count())
print('Unique Airports with FM-15 Report: ', hourly_weather[['STATION','NAME']][hourly_weather['REPORT_TYPE']=='FM-15'].distinct().count())
print('Unique Airports with FM-16 Report: ', hourly_weather[['STATION','NAME']][hourly_weather['REPORT_TYPE']=='FM-16'].distinct().count())
print('Unique Airports with FM-12 Report: ', hourly_weather[['STATION','NAME']][hourly_weather['REPORT_TYPE']=='FM-12'].distinct().count())
print('-'*60)
print('Total Records of FM-15: ', hourly_weather[hourly_weather['REPORT_TYPE']=='FM-15'].count())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Reducing Dataset to Just 'FM-15' Reports

# COMMAND ----------

# reducing dataset to just be the FM-15 Report Type 
hourly_weather = hourly_weather[hourly_weather['REPORT_TYPE'] == 'FM-15']

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 3) Feature Group EDA's

# COMMAND ----------

# reduce dataset to exclude 2019 for EDA
# Create Train and holdout datasets
hourly_weather_EDA = hourly_weather[hourly_weather["DATE"] < '2019-01-01']

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Review of Temperature Features

# COMMAND ----------

# reviewing the temperature based features. 
features = ['HourlyDewPointTemperature',
 'HourlyDryBulbTemperature',
 'HourlyWetBulbTemperature']
funcs.pairplot(hourly_weather_EDA, features)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Review of Pressure Features

# COMMAND ----------

# reviewing pressure features
features = ['HourlyPressureChange',
 'HourlyPressureTendency',
 'HourlySeaLevelPressure',
 'HourlyStationPressure',]
funcs.pairplot(hourly_weather_EDA, features)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC #### Review of Wind Features

# COMMAND ----------

# reviewing wind speed features
features = ['HourlyWindGustSpeed',
 'HourlyWindSpeed']
funcs.pairplot(hourly_weather_EDA, features)

# COMMAND ----------

feature ='HourlyPrecipitation'
funcs.histogram(hourly_weather, feature)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC #### Review of Categorical Features

# COMMAND ----------

display(hourly_weather_EDA.select('HourlySkyConditions').distinct())

# COMMAND ----------

display(hourly_weather_EDA.select('HourlyPresentWeatherType').distinct())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 4) Selection of Hourly Features

# COMMAND ----------

# reviewing the temperature based features. 
selected_features = ['HourlyDryBulbTemperature',
            'HourlyStationPressure',
            'HourlyPressureChange',
            'HourlyWindGustSpeed',
            'HourlyWindDirection',
            'HourlyPrecipitation',
            'HourlyVisibility',
]


# COMMAND ----------

funcs.pairplot(hourly_weather, selected_features)

# COMMAND ----------

hourly_weather = hourly_weather[['STATION', 'DATE','LATITUDE', 'LONGITUDE'] + selected_features ]

display(hourly_weather)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 5) Hourly Observations Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### 5.1) Weather Feature Engineering

# COMMAND ----------

# casting date column to long (number of seconds)
hourly_weather = hourly_weather.withColumn('Date_Long', col('DATE').cast(TimestampType()).cast('long'))

# creating window to partition by Unique Station ID
# and include previous 2 hours of records (equates to 3 hours when including curent)
hours = lambda i: i * 3600


for feature in ['HourlyDryBulbTemperature',
                'HourlyPressureChange',
                'HourlyStationPressure',
                'HourlyWindGustSpeed',
                'HourlyPrecipitation',
                'HourlyVisibility']:
    for i in[3,6,12]:
        window_spec = Window.partitionBy('STATION').orderBy('Date_Long').rangeBetween(-hours(i-1), hours(0))
        hourly_weather = hourly_weather.withColumn(f"{i}Hr_{feature.replace('Hourly', '')}", f.avg(feature).over(window_spec))

display(hourly_weather)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### 5.2) Time Feature Engineering

# COMMAND ----------

# # read in data from parquet
team_blob_url = funcs.blob_connect()
timestamps = spark.read.parquet(f"{team_blob_url}/ES/OTPW/Timestamps")

# joining UTC offsets to each record
hourly_weather = hourly_weather.join(timestamps,
                                   hourly_weather.STATION == timestamps.STATION,
                                   how = 'left') \
                                .drop(timestamps.STATION)

# adding new column for UTC value
hourly_weather = hourly_weather.withColumn('UTC', expr("Date_Long - UTC_offset")) \
                                .withColumn('UTC', col('UTC').cast(TimestampType()))

# drop unnecesary columns
hourly_weather = hourly_weather.drop(*('Date_Long', 'NAME', 'REGION', 'UTC_offset'))

display(hourly_weather)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 6) Write Dataset Back to Blob

# COMMAND ----------

# writing data as parquet to blob
funcs.write_parquet_to_blob(hourly_weather, 'ES/Weather/Updated_Hourly')

# COMMAND ----------

