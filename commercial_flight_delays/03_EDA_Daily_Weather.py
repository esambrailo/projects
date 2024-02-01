# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Daily Weather Summaries Dataset EDA
# MAGIC This notebook is to explore the dataset we created for the daily weather summaries from the original raw dataset. 
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

from pyspark.sql.functions import col
from pyspark.sql import functions as f
from pyspark.sql.types import DateType, TimestampType
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 1) Read-in and address nulls/types

# COMMAND ----------

# # read in data from parquet
team_blob_url = funcs.blob_connect()

daily_weather = spark.read.parquet(f"{team_blob_url}/ES/Weather/SOD")

# grouping features
station_attr = ['STATION', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'NAME', 'SOURCE']
daily_sums = [feature for feature in daily_weather.columns if "Daily" in feature] + ['Sunrise', 'Sunset']
daily_sums.remove('DailyWeather')

# reducing dataframe to relevant columns
daily_weather = daily_weather[['DATE'] + station_attr + daily_sums]

# checking values for each feature
describe = funcs.describe_table(daily_weather)
display(describe)

# COMMAND ----------

# removing trace values
daily_weather = daily_weather.replace('T', '0.005')

# casting daily features as float
for col_name in daily_sums:
    daily_weather = daily_weather.withColumn(col_name, col(col_name).cast('float'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2) Feature Group EDA's
# MAGIC
# MAGIC **EDA was performed on groups of similar features.  For each, a small random sampling (3,000 records) was taken to provide a visual representaionn of the features. The Pearson Correlation Coefficient was also calculated for each.**

# COMMAND ----------

# reduce dataset to exclude 2019 for EDA
# Create Train and holdout datasets
daily_weather_EDA = daily_weather[daily_weather["DATE"] < '2019-01-01']

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Review of Snowfall Features

# COMMAND ----------

# sampling 
features = ['DailySnowfall', 'DailySnowDepth']
funcs.pairplot(daily_weather_EDA, features)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Review of Temperature Features
# MAGIC
# MAGIC "Planes get 1% less lift with every 5.4 degrees Fahrenheit (3 degrees Celsius) of temperature rise." 
# MAGIC
# MAGIC     - https://www.cnn.com/travel/article/climate-change-airplane-takeoff-scn/index.html#:~:text=%E2%80%9CLift%20depends%20on%20several%20factors,of%20temperature%20rise%2C%20Williams%20said.
# MAGIC
# MAGIC Average duration of heat waves over past decade is 4 days. 
# MAGIC
# MAGIC     - https://www.epa.gov/climate-indicators/climate-change-indicators-heat-waves
# MAGIC
# MAGIC "When observing successive weather reports (METARs), a reducing gap between the actual temperature and the Dew Point temperature gives an indication of impending low visibility conditions and the posibility of fog."
# MAGIC
# MAGIC     - https://skybrary.aero/articles/dew-point#:~:text=When%20observing%20successive%20weather%20reports,and%20the%20posibility%20of%20fog.
# MAGIC
# MAGIC The first factor is that airports have runways that are too short for high temperatures. The second is that planes are designed, maintained and safety certified to land and takeoff in a certain temperature range. 
# MAGIC
# MAGIC     - https://www.forbes.com/sites/quora/2017/06/29/why-your-flight-wont-take-off-if-its-too-hot-or-too-cold/?sh=549f93935820 
# MAGIC

# COMMAND ----------

# reviewing the temperature based features. 
features = ['DailyAverageDewPointTemperature',
 'DailyAverageDryBulbTemperature',
 'DailyAverageWetBulbTemperature',
 'DailyDepartureFromNormalAverageTemperature',
 'DailyMaximumDryBulbTemperature',
 'DailyMinimumDryBulbTemperature',
 'DailyCoolingDegreeDays',
 'DailyHeatingDegreeDays',]
funcs.pairplot(daily_weather_EDA, features)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Review of Humidity Features
# MAGIC
# MAGIC "Humidity affects the way an airplane flies because of the change in pressure that accompanies changes in humidity. As the humidity goes up, the air pressure for a given volume of air goes down. This means the wings have fewer air molecules to affect as they are pushed through the airmass. Fewer molecules = less lift. The other problem is that jet engines do not like humidity either. Jet engines are built for cold, dry air, and humid air has fewer oxygen molecules to burn per unit volume. Therefore the engine combusts a little bit less and puts out slightly less thrust."
# MAGIC
# MAGIC     - https://www.physlink.com/education/askexperts/ae652.cfm

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Review of Wind Speed/Direction Features
# MAGIC
# MAGIC "Although wind speed is rarely a problem for commercial airliners, there is a limit to what they can cope with. The main problem is strong crosswinds, that is, horizontal winds approximately at right angles to the direction of takeoff and landing. If these are more than around 35-40 miles per hour, it may be quite difficult for the aircraft to take off, and departure may be delayed for a while."
# MAGIC
# MAGIC     - https://pilotinstitute.com/wind-speed-airplane/#:~:text=The%20only%20thing%20a%20strong,flight%20takes%20longer%20than%20expected.
# MAGIC
# MAGIC "A crosswind above about 40mph and tailwind above 10mph can start to cause problems and stop commercial jets taking off and landing."
# MAGIC
# MAGIC     - https://www.flightdeckfriend.com/ask-a-pilot/aircraft-maximum-wind-limits/#:~:text=A%20crosswind%20above%20about%2040mph,of%20the%20passengers%20and%20crew.

# COMMAND ----------

# reviewing wind speed features
features = ['DailyAverageWindSpeed',
 'DailyPeakWindSpeed',
 'DailySustainedWindSpeed']
funcs.pairplot(daily_weather_EDA, features)

# COMMAND ----------

# reviewing wind direction features
features = ['DailyPeakWindDirection',
 'DailySustainedWindDirection',]
funcs.pairplot(daily_weather_EDA, features)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Review of Air Pressure Features
# MAGIC
# MAGIC 1 inch of mercury = 33.86 millibars
# MAGIC
# MAGIC "Quite simply, a low pressure area is a storm. Hurricanes and large-scale rain and snow events (blizzards and nor'easters) in the winter are examples of storms. Thunderstorms, including tornadoes, are examples of small-scale low pressure areas."
# MAGIC
# MAGIC     - https://www.google.com/search?q=how+do+inches+in+mercury+convert+to+millibars&rlz=1C1JJTC_enUS1020US1020&oq=how+does+inches+in+mercury+convert+to+mil&gs_lcrp=EgZjaHJvbWUqCggHECEYFhgdGB4yBggAEEUYOTIHCAEQIRigATIHCAIQIRigATIHCAMQIRigATIHCAQQIRigATIHCAUQIRirAjIHCAYQIRirAjIKCAcQIRgWGB0YHtIBCjEyNTg5ajBqMTWoAgCwAgA&sourceid=chrome&ie=UTF-8
# MAGIC
# MAGIC     - https://www.worldstormcentral.co/law%20of%20storms/secret%20law%20of%20storms.html#:~:text=A%20storm%20also%20typically%20requires,1009%20hPa%20(or%20mb).
# MAGIC

# COMMAND ----------

# reviewing air pressure
features = ['DailyAverageSeaLevelPressure',
 'DailyAverageStationPressure',]
funcs.pairplot(daily_weather_EDA, features)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Review of Sunrise and Sunset

# COMMAND ----------

# selected daily features to join to flight data
features = ['Sunrise',
            'Sunset',]
funcs.pairplot(daily_weather_EDA, features)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 3) Selected Daily Features

# COMMAND ----------

# selected daily features to join to flight data
selected_features = ['DailySnowfall',
            'DailyPrecipitation',
            'DailyDepartureFromNormalAverageTemperature',
            'DailyAverageDryBulbTemperature',
            'DailyAverageRelativeHumidity',
            'DailyAverageStationPressure',
            'DailySustainedWindDirection',
            'DailySustainedWindSpeed']
funcs.pairplot(daily_weather, selected_features)

daily_weather = daily_weather[['STATION','DATE'] + selected_features ]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 4) Daily Feature Engineering

# COMMAND ----------

# casting date column as datetime
daily_weather = daily_weather.withColumn('DATE', col('DATE').cast(DateType()))

# adding a new column for day after summary
daily_weather = daily_weather.withColumn("Next_Date", f.date_add("DATE", 1))

display(daily_weather)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 5) Write Dataset Back to Blob

# COMMAND ----------

# writing data as parquet to blob
funcs.write_parquet_to_blob(daily_weather, 'ES/Weather/Updated_SOD')