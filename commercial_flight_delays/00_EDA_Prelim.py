# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Phase 1 Preliminary EDA
# MAGIC This workbook is for some preliminary EDA work on both the OTPW joined dataset and the raw weather data.  The underlining purpose is to better understand how the weather was joined in the OTPW dataset. 
# MAGIC
# MAGIC ## 1) Initial Review of Weather Dataset
# MAGIC I am looking at the 3 month subset of the combined dataset and exploring the weather features present.

# COMMAND ----------

# importing modules for smaller summary datasets
import pandas as pd
import numpy as np

# COMMAND ----------

# Reading the raw weather data to retrieve the column names. 
df_weather = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data_3m/")
weather_cols = df_weather.columns
print(f"There are {len(weather_cols)} weather columns in the combined dataset.")

# Reading in the combined dataset and filtering to just weather columns
otpw = spark.read.format("csv").option("header","true").load(f"dbfs:/mnt/mids-w261/OTPW_3M_2015.csv")
df_otpw = otpw[weather_cols]
print(f"The 3 month subsample has {df_otpw.count()} total records.")

# COMMAND ----------

# creating table of feature describe()
feature_sample = df_otpw.describe()

# converting feature sample to pandas df and transposing
feature_sample = feature_sample.toPandas().T

#promoting first row to headers, and dropping row
feature_sample.columns = feature_sample.iloc[0]
feature_sample = feature_sample.drop(feature_sample.index[0])
feature_sample['count'] = pd.to_numeric(feature_sample['count'])

# quantifying a count threshold at 25% total
threshold = 25.0
## since we are referencing a very small sample of data from a specific time period
## I do not what to put too much weight on how many nulls are present in this sample
## that is why I have the threshold at 25%. 
cnt_threshold = max(feature_sample['count'])*(threshold/100.0)
print(f"Count threshold is {cnt_threshold}.")

# displaying records that are being removed
print(f"Below are the columns with less than {threshold}% records present:")
potential_drops = feature_sample['count'][feature_sample['count'] < cnt_threshold]
potential_drops = pd.DataFrame(potential_drops).reset_index()
display(potential_drops)

# COMMAND ----------

# based on a review of the features we are looking to omit due to limited records, 
# there are a series of 'Daily' metrics that could still be relevant and may not make the threshold 
# because they are only captured daily (1/24 < 25%).  Planning to re-include those in the list. 

# quantifying a list of the 'Daily' features
daily_cols = []
for column in potential_drops['index']:
    if "Daily" in column:
        daily_cols.append(column)

# reviewing remaining feature list, sorted by count
updated_features = feature_sample[(feature_sample['count'] >= cnt_threshold) | (feature_sample.index.isin(daily_cols))]
updated_features = updated_features.sort_values(by='count', ascending=False)
print(f"There are {updated_features.shape[0]} features.")
updated_features

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Specific Exploration of Raw Weather Data

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 1) Number of Weather Stations Present

# COMMAND ----------

# count of unique weather stations
df_weather.select('STATION').distinct().count()

# COMMAND ----------

# selecting a specific day and location to see all observations recorded for it
example_data = df_weather[(df_weather['STATION'] =='72295023174') & (df_weather['DATE'].contains('2015-01-10'))]
display(example_data.sort('REPORT_TYPE'))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 3) Checking if Weather Data is Always Departing Location

# COMMAND ----------

weather_loc_check = otpw[otpw['STATION']!=otpw['origin_station_id']]
display(weather_loc_check)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 4) Reviewing Time-Based Components of Joined Dataset

# COMMAND ----------

lax_join_review = otpw['sched_depart_date_time', 'sched_depart_date_time_UTC', 'four_hours_prior_depart_UTC', 'two_hours_prior_depart_UTC', 'DATE', 'Report_Type'][otpw['STATION']=='72295023174']
display(lax_join_review)