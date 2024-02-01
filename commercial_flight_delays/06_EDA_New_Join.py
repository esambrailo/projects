# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Weather-Related EDA for Newly Joined Flight and Weather Dataset
# MAGIC This notebook is to review, explore the newly created joined flight and weather dataset. 
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
# MAGIC ### 1) Read-in and group weather features

# COMMAND ----------

# # read in data from parquet
team_blob_url = funcs.blob_connect()

new_df = spark.read.parquet(f"{team_blob_url}/LH/1yr_clean_temp")
# new_df = new_df[new_df['FL_DATE'] < '2019-01-01']

# assign target features
target_metric = 'DEP_DELAY_NEW'
target_classifier = 'DEP_DEL15'

# grouping features
origin_daily_attr = ['origin_DailySnowfall',
                      'origin_DailyPrecipitation',
                      'origin_DailyDepartureFromNormalAverageTemperature',
                      'origin_DailyAverageDryBulbTemperature',
                      'origin_DailyAverageRelativeHumidity',
                      'origin_DailyAverageStationPressure',
                      'origin_DailySustainedWindDirection',
                      'origin_DailySustainedWindSpeed']
origin_hourly_attr = ['origin_HourlyDryBulbTemperature',
                      'origin_HourlyStationPressure',
                      'origin_HourlyPressureChange',
                      'origin_HourlyWindGustSpeed',
                      'origin_HourlyWindDirection',
                      'origin_HourlyPrecipitation',
                      'origin_HourlyVisibility',
                      'origin_3Hr_DryBulbTemperature',
                      'origin_6Hr_DryBulbTemperature',
                      'origin_12Hr_DryBulbTemperature',
                      'origin_3Hr_PressureChange',
                      'origin_6Hr_PressureChange',
                      'origin_12Hr_PressureChange',
                      'origin_3Hr_StationPressure',
                      'origin_6Hr_StationPressure',
                      'origin_12Hr_StationPressure',
                      'origin_3Hr_WindGustSpeed',
                      'origin_6Hr_WindGustSpeed',
                      'origin_12Hr_WindGustSpeed',
                      'origin_3Hr_Precipitation',
                      'origin_6Hr_Precipitation',
                      'origin_12Hr_Precipitation',
                      'origin_3Hr_Visibility',
                      'origin_6Hr_Visibility',
                      'origin_12Hr_Visibility']
dest_daily_attr = ['dest_DailySnowfall',
                  'dest_DailyPrecipitation',
                  'dest_DailyDepartureFromNormalAverageTemperature',
                  'dest_DailyAverageDryBulbTemperature',
                  'dest_DailyAverageRelativeHumidity',
                  'dest_DailyAverageStationPressure',
                  'dest_DailySustainedWindDirection',
                  'dest_DailySustainedWindSpeed']
dest_hourly_attr = ['dest_HourlyDryBulbTemperature',
                    'dest_HourlyStationPressure',
                    'dest_HourlyPressureChange',
                    'dest_HourlyWindGustSpeed',
                    'dest_HourlyWindDirection',
                    'dest_HourlyPrecipitation',
                    'dest_HourlyVisibility',
                    'dest_3Hr_DryBulbTemperature',
                    'dest_6Hr_DryBulbTemperature',
                    'dest_12Hr_DryBulbTemperature',
                    'dest_3Hr_PressureChange',
                    'dest_6Hr_PressureChange',
                    'dest_12Hr_PressureChange',
                    'dest_3Hr_StationPressure',
                    'dest_6Hr_StationPressure',
                    'dest_12Hr_StationPressure',
                    'dest_3Hr_WindGustSpeed',
                    'dest_6Hr_WindGustSpeed',
                    'dest_12Hr_WindGustSpeed',
                    'dest_3Hr_Precipitation',
                    'dest_6Hr_Precipitation',
                    'dest_12Hr_Precipitation',
                    'dest_3Hr_Visibility',
                    'dest_6Hr_Visibility',
                    'dest_12Hr_Visibility']


# # checking values for each feature
describe = funcs.describe_table(new_df)
display(describe)

# COMMAND ----------

new_df.columns

# COMMAND ----------

# removing null values from DEP_DEL15
new_df = new_df[new_df[target_classifier].isNotNull()]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 2) Feature Group EDA's

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Functions to Move to Funcs

# COMMAND ----------

def pandas_sampling(df, sample_size=3000):
    # quantifying total records
    total_records = df.count()

    # determining sampling fraction
    if sample_size > total_records:
        fraction = 1.0
    else:
        fraction = sample_size/total_records
    print("Pandas Sample Taken")
    print('-'*60)
    print("Total records: ", total_records)
    print("Records sampled: ", total_records*fraction)
    print("Fraction of total records: ", fraction)
    
    # sampling df for pandas visualizations
    data_sample = df.sample(fraction = fraction).toPandas()

    return data_sample

def box_plot(feature, target, df, labels, sample_size=3000, label_mapping = {1: 'Delayed', 0: 'On-Time'}):
    df = pandas_sampling(df, sample_size)
    
    # Modify the figure size
    plt.figure(figsize=(4, 8))

    # Set the faint background color
    plt.rcParams['axes.facecolor'] = '#f5f5f5'

    grouped_data = []
    categories = df[target].unique()

    for category in categories:
        grouped_data.append(df[df[target] == category][feature])

    # Plotting the box plot
    boxplot = plt.boxplot(grouped_data, vert=True, patch_artist=True, labels=[label_mapping[category] for category in categories])

    box_color = 'white'
    for box in boxplot['boxes']:
        box.set(facecolor=box_color)

    plt.ylabel(labels[2])
    plt.xlabel(labels[1])
    plt.title(labels[0])
    plt.xticks(rotation=45, ha="right")

    # Set the background color with a faint version of '#c78a2e'
    ax = plt.gca()
    ax.set_facecolor('#f5f5f5')

    # Add count annotations to the boxplot
    counts = [len(data) for data in grouped_data]
    positions = range(1, len(categories) + 1)
    for pos, count in zip(positions, counts):
        ax.annotate(str(count), xy=(pos, max(ax.get_ylim())), xytext=(0, -15),
                        textcoords="offset points", ha='center', va='bottom')

    plt.show()

def feature_threshold_comparison(df, review_feature, thresholds, target_classifier, impute=False, inverse=False):
    '''Creates a pandas table and markdown printout for the number of records 
    that exist for both delayed and ontime flights based on the assigned thresholds
    for the feature being reviewed.'''

    # impute feature nulls with 0 or filter out nulls
    if impute:
        temp_df = df.fillna(0, subset=[review_feature])
    else:
        temp_df = df[df[review_feature].isNotNull()]
    
    # initiating table for results
    table = pd.DataFrame(columns=['Threshold', 'Delayed Flight Count', 'On Time Flight Count', 'Percent Delayed'])
    
    # iterating through df for each treshold and capturing results. To reduce compute, iterations use previous iter's df so thresholds
    # must be sequential and in a reducing direction. 
    for thresh in thresholds:
        if inverse:
            temp_df = temp_df[temp_df[review_feature] <= thresh].cache()
        else:
            temp_df = temp_df[temp_df[review_feature] >= thresh].cache()
        # total count
        count = temp_df.count()
        # delayed count
        delayed = temp_df[temp_df[target_classifier] == 1].count()
        # ontime count
        ontime = count - delayed
        # percentage calc
        percent = round(delayed/count, 2)
        # append to table
        table.loc[len(table)] = [thresh, delayed, ontime, percent]

    # create a markdown version of table
    markdown_table = table.to_markdown(index=False)
    table_title = f"Review of flight delay counts \n at various thresholds for '{review_feature}'"
    markdown_table = f"**{table_title}**\n\n{markdown_table}"
    display(table)
    print()
    print(markdown_table)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Review of Snow Features

# COMMAND ----------

# list of thresholds to review. MUST BE IN ASCENDING ORDER!
thresholds = [0, .1, 1, 6, 12]
feature_threshold_comparison(new_df, 'origin_DailySnowfall', thresholds, target_classifier, impute=True)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Review of flight delay counts 
# MAGIC  at various thresholds for 'origin_DailySnowfall'**
# MAGIC
# MAGIC |   Threshold |   Delayed Flight Count |   On Time Flight Count |   Percent Delayed |
# MAGIC |------------:|-----------------------:|-----------------------:|------------------:|
# MAGIC |         0   |            1.05578e+06 |             4.6704e+06 |              0.18 |
# MAGIC |         0.1 |        48592           |        111201          |              0.3  |
# MAGIC |         1   |        25573           |         48258          |              0.35 |
# MAGIC |         6   |         3637           |          3691          |              0.5  |
# MAGIC |        12   |         1047           |           510          |              0.67 |

# COMMAND ----------

# Impute snowfall nulls to 0
snow_df = new_df.fillna(0, subset=['origin_DailySnowfall'])

# reviewing all records
labels = ["Previous Day's Snowfall for Delayed & On-Time Flights", 
          'Depature', 
          'Snowfall at Departure Airport (Prev. Day)']

box_plot('origin_DailySnowfall', target_classifier, snow_df, labels)

# COMMAND ----------

# reducing df to just recorded snow
snow1_df = snow_df[snow_df['origin_DailySnowfall'] > 1]

# reviewing all records
labels = ['Prev. Day Snowfall > 1" for Delayed & On-Time Flights', 
          'Depature', 
          'Snowfall at Departure Airport (Prev. Day)']

box_plot('origin_DailySnowfall', target_classifier, snow1_df, labels)

# COMMAND ----------

# reducing df to just recorded snow
snow6_df = snow_df[snow_df['origin_DailySnowfall'] > 6]

# reviewing all records
labels = ['Prev. Day Snowfall > 6" for Delayed & On-Time Flights', 
          'Depature', 
          'Snowfall at Departure Airport (Prev. Day)']

box_plot('origin_DailySnowfall', target_classifier, snow6_df, labels)

# COMMAND ----------

# Impute snowfall nulls to 0
snow_df = new_df.fillna(0, subset=['dest_DailySnowfall'])

# reducing df to just recorded snow
snow6_df = snow_df[snow_df['dest_DailySnowfall'] > 6]

# reviewing all records
labels = ['Prev. Day Snowfall > 6" for Delayed & On-Time Flights', 
          'Depature', 
          'Snowfall at Arrival Airport (Prev. Day)']

box_plot('dest_DailySnowfall', target_classifier, snow6_df, labels)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Review of Precipitation Features

# COMMAND ----------

# list of thresholds to review. MUST BE IN ASCENDING ORDER!
thresholds = [0, 1, 2, 4, 6]
feature_threshold_comparison(new_df, 'origin_DailyPrecipitation', thresholds, target_classifier, impute=True)

# COMMAND ----------

# MAGIC %md
# MAGIC **Review of flight delay counts 
# MAGIC  at various thresholds for 'origin_DailyPrecipitation'**
# MAGIC
# MAGIC |   Threshold |   Delayed Flight Count |   On Time Flight Count |   Percent Delayed |
# MAGIC |------------:|-----------------------:|-----------------------:|------------------:|
# MAGIC |           0 |            1.05578e+06 |             4.6704e+06 |              0.18 |
# MAGIC |           1 |        40389           |        118019          |              0.25 |
# MAGIC |           2 |         9868           |         25663          |              0.28 |
# MAGIC |           4 |         1045           |          2901          |              0.26 |
# MAGIC |           6 |           97           |           324          |              0.23 |

# COMMAND ----------

review_features = ['dest_HourlyPrecipitation', 'dest_3Hr_Precipitation', 'dest_6Hr_Precipitation', 'dest_12Hr_Precipitation']
# list of thresholds to review. MUST BE IN ASCENDING ORDER!
thresholds = [0, .1, .2, .4, .8]
for feature in review_features:
    feature_threshold_comparison(new_df, feature, thresholds, target_classifier, impute=True)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Review of flight delay counts 
# MAGIC  at various thresholds for 'dest_12Hr_Precipitation'**
# MAGIC
# MAGIC |   Observed Precipitation Threshold |   Percent Delayed | Delayed Flight Count |   On Time Flight Count | 
# MAGIC |------------:|-----------------------:|-----------------------:|------------------:|
# MAGIC |       > 0"  |           18% |       1.05578e+06 |             4.6704e+06 |        
# MAGIC |      > 0.1" |          27% |    11231           |         30735          |       
# MAGIC |      > 0.2" |          26% |     2129           |          6092          |       
# MAGIC |      > 0.4" |          31% |      221           |           493          |      
# MAGIC |      > 0.8" |          56% |      24           |            19          |       

# COMMAND ----------

# reducing df to just recorded snow
prep_df = new_df[new_df['dest_3Hr_Precipitation'] > 0.2]

# reviewing all records
labels = ['Avg. Precipitation (over 3 hours)', 
          'Depature', 
          '3Hr Average Precipitation at Arrival Airport (3 Hours Prior)']

box_plot('origin_DailySnowfall', target_classifier, prep_df, labels)

# COMMAND ----------

# feature reviewing
review_feature = 'origin_HourlyPrecipitation'

# Impute precipitation nulls to 0
prec_df = new_df.fillna(0, subset=[review_feature])

# # reducing df to just recorded snow
# prec_df = prec_df[prec_df['dest_DailySnowfall'] > 6]

# reviewing all records
labels = ['Hourly Precipitation for Delayed & On-Time Flights', 
          'Depature', 
          'Precipitation at Departure Airport ( Approx. 3 Hrs Prior)']

box_plot(review_feature, target_classifier, prec_df, labels)

# COMMAND ----------

# reducing df to just recorded snow
prec2_df = prec_df[prec_df[review_feature] > 0.2]

# reviewing all records
labels = ['Hourly Precipitation > 0.2" for Delayed & On-Time Flights', 
          'Depature', 
          'Precipitation at Departure Airport ( Approx. 3 Hrs Prior)']

box_plot(review_feature, target_classifier, prec2_df, labels)

# COMMAND ----------

# reducing df to just recorded snow
prec2_df = prec_df[prec_df['origin_DailyPrecipitation'] > 2]

# reviewing all records
labels = ['Prev. Day Precipitation > 2" \nfor Delayed & On-Time Flights', 
          'Depature', 
          'Precipitation at Departure Airport (Prev. Day)']

box_plot('origin_DailyPrecipitation', target_classifier, prec2_df, labels)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Visibility Features

# COMMAND ----------

review_features = ['origin_HourlyVisibility',
                'origin_3Hr_Visibility',
                'origin_6Hr_Visibility',
                'origin_12Hr_Visibility']
# list of thresholds to review. MUST BE IN ASCENDING ORDER!
thresholds = [10, 6, 4, 2]
for feature in review_features:
    feature_threshold_comparison(new_df, feature, thresholds, target_classifier, impute=False, inverse=True)

# COMMAND ----------

# MAGIC %md
# MAGIC **Review of flight delay counts 
# MAGIC  at various thresholds for 'origin_12Hr_Visibility'**
# MAGIC
# MAGIC |   Threshold |   Delayed Flight Count |   On Time Flight Count |   Percent Delayed |
# MAGIC |------------:|-----------------------:|-----------------------:|------------------:|
# MAGIC |          10 |            1.05437e+06 |            4.66126e+06 |              0.18 |
# MAGIC |           6 |        95283           |       244996           |              0.28 |
# MAGIC |           4 |        45503           |       104187           |              0.3  |
# MAGIC |           2 |        14000           |        30991           |              0.31 |

# COMMAND ----------

# reducing df to just recorded snow
prep_df = new_df[new_df['origin_12Hr_Visibility'] < 6]

# reviewing all records
labels = ['Avg. Precipitation (over 3 hours)', 
          'Depature', 
          '3Hr Average Precipitation at Arrival Airport (3 Hours Prior)']

box_plot('origin_DailySnowfall', target_classifier, prep_df, labels)

# COMMAND ----------

