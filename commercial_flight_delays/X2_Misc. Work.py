# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC # Notebook of Misc. Work
# MAGIC
# MAGIC This notebook is to house any misc. work completed/explored. 

# COMMAND ----------

# importing custom functions
import funcs

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## 1) Exploring write partitioning

# COMMAND ----------

from pyspark.sql import functions as f
from pyspark.sql.window import Window
from pyspark.sql.types import  TimestampType

# the 261 course blob storage is mounted here.
mids261_mount_path      = "/mnt/mids-w261"
display(dbutils.fs.ls(f"{mids261_mount_path}/"))

# COMMAND ----------

# read in daily weather data from parquet
team_blob_url = funcs.blob_connect()
display(dbutils.fs.ls(f"{team_blob_url}/ES/RF/"))

# COMMAND ----------

def write_parquet_to_blob(df, location, partition):
    '''
    This function writes a dataframe to our team's blob storage
    at the location passed in as an argument.
    '''
    # connect to blob
    team_blob_url = funcs.blob_connect()

    # write to blob
    df.write.partitionBy(partition).mode('overwrite').parquet(f"{team_blob_url}/{location}")

write_parquet_to_blob(hourly_weather, 'ES/Weather/Updated_Hourly_test', 'STATION')


# COMMAND ----------

hourly_weather = spark.read.parquet(f"{team_blob_url}/ES/Weather/Updated_Hourly_test")
display(hourly_weather)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 2) Creating Back-up Copy of Large Joined Dataset

# COMMAND ----------

# # read in data from parquet
team_blob_url = funcs.blob_connect()
df = spark.read.parquet(f"{team_blob_url}/ES/new_joins/5Y")

# COMMAND ----------

# writing data as parquet to blob
funcs.write_parquet_to_blob(df, 'Back_up/new_joins/5Y')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 3) Casting Full Dataset Types & Holding-out 2019

# COMMAND ----------

# MAGIC %md
# MAGIC #### Casting

# COMMAND ----------

# # read in data from parquet
team_blob_url = funcs.blob_connect()
df = spark.read.parquet(f"{team_blob_url}/ES/new_joins/5Y")

columns = df.columns
print(f"There are {len(columns)} columns in the 5Y dataset.")

# COMMAND ----------

# define function to cast spark df columsn to new type(s) and apply to 3-month dataset
def cast_cols_to_new_type(spark_df, columns_to_cast):
    # define the expressions for each column
    cast_exprs = [f"cast({col} as {new_type}) as {col}" for col, new_type in columns_to_cast.items()]
    # apply the cast expressions
    spark_df = spark_df.selectExpr(*cast_exprs)
    #spark_df.printSchema()
    return spark_df

# create dictionary of ALL column types 
# (note: casting only selected cols to change will omit them from resulting df)
new_coltypes_dict = {'QUARTER': 'integer', 
                     'DAY_OF_MONTH': 'integer', 
                     'DAY_OF_WEEK': 'integer',
                     'FL_DATE': 'date',
                     'OP_UNIQUE_CARRIER': 'string',
                     'OP_CARRIER_AIRLINE_ID': 'integer',
                     'OP_CARRIER': 'string', 
                     'TAIL_NUM': 'string', 
                     'OP_CARRIER_FL_NUM': 'string',
                     'ORIGIN_AIRPORT_ID': 'string',
                     'ORIGIN_AIRPORT_SEQ_ID': 'string',
                     'ORIGIN_CITY_MARKET_ID': 'string',
                     'ORIGIN': 'string',
                     'ORIGIN_CITY_NAME': 'string',
                     'ORIGIN_STATE_ABR': 'string',
                     'ORIGIN_STATE_FIPS': 'integer',
                     'ORIGIN_STATE_NM': 'string',
                     'ORIGIN_WAC': 'integer',
                     'DEST_AIRPORT_ID': 'string',
                     'DEST_AIRPORT_SEQ_ID': 'integer',
                     'DEST_CITY_MARKET_ID': 'integer',
                     'DEST': 'string',
                     'DEST_CITY_NAME': 'string',
                     'DEST_STATE_ABR': 'string', 
                     'DEST_STATE_FIPS': 'integer',
                     'DEST_STATE_NM': 'string',
                     'DEST_WAC': 'integer',
                     'CRS_DEP_TIME': 'float', 
                     'DEP_TIME': 'float',  
                     'DEP_DELAY': 'double',
                     'DEP_DELAY_NEW': 'double', 
                     'DEP_DEL15': 'double', 
                     'DEP_DELAY_GROUP': 'integer',
                     'DEP_TIME_BLK': 'string', 
                     'TAXI_OUT': 'double',
                     'WHEELS_OFF': 'float',
                     'WHEELS_ON': 'float',
                     'TAXI_IN': 'double',
                     'CRS_ARR_TIME': 'integer',
                     'ARR_TIME': 'integer',
                     'ARR_DELAY': 'double',
                     'ARR_DELAY_NEW': 'double',
                     'ARR_DEL15': 'double',
                     'ARR_DELAY_GROUP': 'integer',
                     'ARR_TIME_BLK': 'string',
                     'CANCELLED': 'double',
                     'CANCELLATION_CODE': 'string',
                     'DIVERTED': 'double',
                     'CRS_ELAPSED_TIME': 'double',
                     'ACTUAL_ELAPSED_TIME': 'double',
                     'AIR_TIME': 'double',
                     'FLIGHTS': 'float', 
                     'DISTANCE': 'double',
                     'DISTANCE_GROUP': 'integer',
                     'CARRIER_DELAY': 'double',
                     'WEATHER_DELAY': 'double', 
                     'NAS_DELAY': 'double',
                     'SECURITY_DELAY': 'double',
                     'LATE_AIRCRAFT_DELAY': 'double',
                     'FIRST_DEP_TIME': 'integer',
                     'TOTAL_ADD_GTIME': 'double',
                     'LONGEST_ADD_GTIME': 'double',
                     'YEAR': 'integer',
                     'MONTH': 'integer',
                     'origin_airport_name': 'string',
                     'origin_station_name': 'string',
                     'origin_station_id': 'bigint',
                     'origin_iata_code': 'string',
                     'origin_icao': 'string',
                     'origin_type': 'string',
                     'origin_region': 'string',
                     'origin_station_lat': 'double', 
                     'origin_station_lon': 'double', 
                     'origin_airport_lat': 'double',
                     'origin_airport_lon': 'double',
                     'origin_station_dis': 'double',
                     'dest_airport_name': 'string',
                     'dest_station_name': 'string',
                     'dest_station_id': 'bigint',
                     'dest_iata_code': 'string',
                     'dest_icao': 'string',
                     'dest_type': 'string',
                     'dest_region': 'string',
                     'dest_station_lat': 'double', 
                     'dest_station_lon': 'double', 
                     'dest_airport_lat': 'double',
                     'dest_airport_lon': 'double',
                     'dest_station_dis': 'double',
                     'sched_depart_date_time_UTC': 'timestamp',
                     'four_hours_prior_depart_UTC': 'timestamp',
                     'two_hours_prior_depart_UTC': 'timestamp',
                     'origin_DailySnowfall': 'float',
                     'origin_DailyPrecipitation': 'float',
                     'origin_DailyDepartureFromNormalAverageTemperature': 'float',
                     'origin_DailyAverageDryBulbTemperature': 'float',
                     'origin_DailyAverageRelativeHumidity': 'float',
                     'origin_DailyAverageStationPressure': 'float',
                     'origin_DailySustainedWindDirection': 'float',
                     'origin_DailySustainedWindSpeed': 'float',
                     'dest_DailySnowfall': 'float',
                     'dest_DailyPrecipitation': 'float',
                     'dest_DailyDepartureFromNormalAverageTemperature': 'float',
                     'dest_DailyAverageDryBulbTemperature': 'float',
                     'dest_DailyAverageRelativeHumidity': 'float',
                     'dest_DailyAverageStationPressure': 'float',
                     'dest_DailySustainedWindDirection': 'float',
                     'dest_DailySustainedWindSpeed': 'float',
                     'three_hours_prior_depart_UTC': 'timestamp',
                     'origin_DATE': 'date',
                     'origin_HourlyDryBulbTemperature': 'float',
                     'origin_HourlyStationPressure': 'float',
                     'origin_HourlyPressureChange': 'float',
                     'origin_HourlyWindGustSpeed': 'float',
                     'origin_HourlyWindDirection': 'float',
                     'origin_HourlyPrecipitation': 'float',
                     'origin_HourlyVisibility': 'float',
                     'origin_3Hr_DryBulbTemperature': 'double',
                     'origin_6Hr_DryBulbTemperature': 'double',
                     'origin_12Hr_DryBulbTemperature': 'double',
                     'origin_3Hr_PressureChange': 'double',
                     'origin_6Hr_PressureChange': 'double',
                     'origin_12Hr_PressureChange': 'double',
                     'origin_3Hr_StationPressure': 'double',
                     'origin_6Hr_StationPressure': 'double',
                     'origin_12Hr_StationPressure': 'double',
                     'origin_3Hr_WindGustSpeed': 'double',
                     'origin_6Hr_WindGustSpeed': 'double',
                     'origin_12Hr_WindGustSpeed': 'double',
                     'origin_3Hr_Precipitation': 'double',
                     'origin_6Hr_Precipitation': 'double',
                     'origin_12Hr_Precipitation': 'double',
                     'origin_3Hr_Visibility': 'double',
                     'origin_6Hr_Visibility': 'double',
                     'origin_12Hr_Visibility': 'double',
                     'origin_UTC': 'timestamp',
                     'dest_DATE': 'date',
                     'dest_HourlyDryBulbTemperature': 'float',
                     'dest_HourlyStationPressure': 'float',
                     'dest_HourlyPressureChange': 'float',
                     'dest_HourlyWindGustSpeed': 'float',
                     'dest_HourlyWindDirection': 'float',
                     'dest_HourlyPrecipitation': 'float',
                     'dest_HourlyVisibility': 'float',
                     'dest_3Hr_DryBulbTemperature': 'double',
                     'dest_6Hr_DryBulbTemperature': 'double',
                     'dest_12Hr_DryBulbTemperature': 'double',
                     'dest_3Hr_PressureChange': 'double',
                     'dest_6Hr_PressureChange': 'double',
                     'dest_12Hr_PressureChange': 'double',
                     'dest_3Hr_StationPressure': 'double',
                     'dest_6Hr_StationPressure': 'double',
                     'dest_12Hr_StationPressure': 'double',
                     'dest_3Hr_WindGustSpeed': 'double',
                     'dest_6Hr_WindGustSpeed': 'double',
                     'dest_12Hr_WindGustSpeed': 'double',
                     'dest_3Hr_Precipitation': 'double',
                     'dest_6Hr_Precipitation': 'double',
                     'dest_12Hr_Precipitation': 'double',
                     'dest_3Hr_Visibility': 'double',
                     'dest_6Hr_Visibility': 'double',
                     'dest_12Hr_Visibility': 'double',
                     'dest_UTC': 'timestamp'}


# comparing columns of dataset and dictionary of columns identified to cast. 
print(f"There are {len(new_coltypes_dict.keys())} columns identified for casting.")
print("Missing from dataset are:")
for col in columns:
    if col not in new_coltypes_dict.keys():
        print(col)
print("")
print("Missing from indentified columns are:")
for col in new_coltypes_dict.keys():
    if col not in columns:
        print(col)

# COMMAND ----------

# applying casting to columns
df_schema = cast_cols_to_new_type(df, new_coltypes_dict)

# checking casting of columns
df_schema.printSchema()

# COMMAND ----------

# writing data as parquet to blob
funcs.write_parquet_to_blob(df, 'ES/new_joins/5YR_schema')

# COMMAND ----------

# reviewing min and max date values of the 5 Year Dataset
def min_max_values(df, column):
    '''Prints min and max values of a specified df and column.'''
    min_value = df.agg({column: "min"}).collect()[0][0]
    max_value = df.agg({column: "max"}).collect()[0][0]
    print(f"Minimum value: {min_value}")
    print(f"Maximum value: {max_value}")

min_max_values(df, 'FL_DATE')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Holdout

# COMMAND ----------

# Create Train and holdout datasets
train = df[df["FL_DATE"] < '2019-01-01']

# writing data as parquet to blob
funcs.write_parquet_to_blob(df, 'ES/train_schema')

# COMMAND ----------

# Create Train and holdout datasets
test = df[df["FL_DATE"] >= '2019-01-01']

# writing data as parquet to blob
funcs.write_parquet_to_blob(df, 'ES/test_schema')

# COMMAND ----------

print("Train Dataset:")
min_max_values(train, 'FL_DATE')
print("Test Dataset:")
min_max_values(test, 'FL_DATE')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 4) Check a parquet file

# COMMAND ----------

# # read in data from parquet
team_blob_url = funcs.blob_connect()
df = spark.read.parquet(f"{team_blob_url}/ES/RF/Model3_val_train/")
display(df)

# COMMAND ----------

df.count()

# COMMAND ----------

# writing data as parquet to blob
funcs.write_parquet_to_blob(df, 'ES/RF/Model3_long_val/')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 4) Deriving Test Split Percentage
# MAGIC
# MAGIC Reviewing the number of records in the final dataset to capture the percentage split that separates 2019. 

# COMMAND ----------

# # read in data from parquet
team_blob_url = funcs.blob_connect()
df = spark.read.parquet(f"{team_blob_url}/BK/clean_5yr_eng")

# COMMAND ----------

train_count = df[df['sched_depart_date_time_UTC'] < '2019-01-01'].count()
test_count = df[df['sched_depart_date_time_UTC'] >= '2019-01-01'].count()

# COMMAND ----------

print("train: ", train_count)
print("test: ", test_count)

print("train percent:", train_count/(train_count+test_count))