# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Splitting Raw Weather Data
# MAGIC This notebook is being used for reviewing and splitting the raw weather dataset into the various report types. 
# MAGIC

# COMMAND ----------

# importing custom functions
import funcs

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 1) Read-in Raw Dataset
# MAGIC The following section is importing modules, reading in data and providing high level visibility to the dataset attributes

# COMMAND ----------

# reviewing the project blob storage
mids261_mount_path      = "/mnt/mids-w261"
display(dbutils.fs.ls(f"{mids261_mount_path}/datasets_final_project_2022/"))

# COMMAND ----------

# Reading the 3 month raw weather dataset. 
weather = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data/")

# collecting all the columns
weather_cols = weather.columns
print(f"There are {len(weather_cols)} weather columns in the combined dataset.")

print("List of Weather Columns:")
weather_cols

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Reduce to Relevant Stations
# MAGIC We are reducing the weather data to just the stations that were mapped to either destination or arriving airport locations in the OTWP dataset.

# COMMAND ----------

# connect to blob
team_blob_url = funcs.blob_connect()

# Reading the relevant weather stations & converting to list
stations = spark.read.parquet(f"{team_blob_url}/ES/OTPW/All_Stations/")
stations = list(stations.select('STATION').toPandas()['STATION'])

# reducing weather to only relevant stations
weather = weather[weather['STATION'].isin(stations)]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Split by Report Type
# MAGIC Based off preliminary EDA in the prior phase, and by review of the documentation, it was apparent that this dataset is truely an aggregations of multiple datasets: 
# MAGIC  - periodic observation reporting (15 min., hourly, etc) 
# MAGIC  - aggregated observation summaries (daily, monthly) 
# MAGIC  It did not appear that any record supported both of these, it was either for one or the other.  For that reason, we have chosen to segregate the dataset based on the report type.

# COMMAND ----------

# reviewing the report types present. 
display(weather.select('REPORT_TYPE').distinct())

# COMMAND ----------

# splitting dataset by report type
df_SOD, df_SOM, df = funcs.split_by_report_type(weather)

# dropping all the empty columns in each df
df_SOD = funcs.drop_empty_cols(df_SOD)
df_SOM = funcs.drop_empty_cols(df_SOM)
df = funcs.drop_empty_cols(df)

# writing data as parquet to blob
funcs.write_parquet_to_blob(df_SOM, 'ES/Weather/SOM')
funcs.write_parquet_to_blob(df_SOD, 'ES/Weather/SOD')
funcs.write_parquet_to_blob(df, 'ES/Weather/Hourly')

print("Number of Records by Report")
print("Daily: ", df_SOD.count())
print("Monthly: ", df_SOM.count())
print("Hourly/Other: ", df.count())

# COMMAND ----------

# see what's in the blob storage root folder 
# connect to blob
team_blob_url = funcs.blob_connect()
display(dbutils.fs.ls(f"{team_blob_url}/ES/Weather/SOM/"))

# # delete directories from blob
# dbutils.fs.rm(f"{team_blob_url}/ES/Weather/SOM/", True)