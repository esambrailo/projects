# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Decision Tree Model
# MAGIC Updated to include categorical features, 
# MAGIC cross-validation, 
# MAGIC and weighted class training for class imbalance

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importing Modules

# COMMAND ----------

# importing custom functions
import funcs

# in memory df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Spark
from pyspark.sql.functions import col, percent_rank, when, udf, array, sum as _sum
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.window import Window
from pyspark.sql.types import ArrayType, IntegerType, DoubleType, StructType, StructField, StringType, ByteType

# DT model
from pyspark.ml.classification import DecisionTreeClassifier

# xgboost
from xgboost.spark import SparkXGBClassifier

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Reading in Datasets

# COMMAND ----------

# read in daily weather data from parquet
team_blob_url = funcs.blob_connect()
df = spark.read.parquet(f"{team_blob_url}/BK/clean_5yr_WITHOUT_2019").cache()

df = df[df['sched_depart_date_time_UTC'] < '2019-01-01']

# COMMAND ----------

df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Defining Functions
# MAGIC Current placeholder for functions for this notebook

# COMMAND ----------

def ohe_features(df, features):
    ''' 
    Takes in a spark dataframe and categorical features.
    Converts each categorical feature to index values and then OHE's them.
    Drops the original column and returns a dataframe with the OHE vector.
    Also returns a list of the OHE features created.
    '''
    # initiating list of features
    ohe_features = []
    # iterating through each categorical feature
    for feature in features:
        # creating an index for values of categorical feature
        indexer = StringIndexer(inputCol=feature, outputCol=f"{feature}_index")
        # creating an encoder
        encoder = OneHotEncoder(inputCol=f"{feature}_index", outputCol=f"{feature}_encoded")
        # developing pipeline
        pipeline = Pipeline(stages=[indexer, encoder])
        # fitting model to indexer then encoder
        model = pipeline.fit(df)
        # transforming df to fit
        df = model.transform(df)
        # dropping unnecessary columns
        df = df.drop(*[feature, f"{feature}_index"])
        # adding new feature to list
        ohe_features.append(f"{feature}_encoded")
    return df, ohe_features

def cross_val_percentages(num_blocks=5, split_ratio=0.8):
    '''
    Creates cross validation block percentiles for both the train and validation sets
    based off the number of blocks and split ratios identified.
    '''
    # creating percentile boundaries for train and validation blocks
    val_area = 1- (1-split_ratio) * 1/num_blocks
    train_block = (1-split_ratio) * 1/num_blocks
    train_blocks_boundaries = [(val_area*i/num_blocks, val_area*(i+1)/num_blocks) for i in range(num_blocks)]
    val_blocks_boundaries = [(val_block[1], val_block[1] + train_block ) for val_block in train_blocks_boundaries]
    print("Train blocks: ", train_blocks_boundaries)
    print("Validation blocks: ", val_blocks_boundaries)
    return train_blocks_boundaries, val_blocks_boundaries

def create_validation_blocks(df, split_feature, blocks=5, split=0.8):
    '''
    Function that orders and ranks a df based on a specified feature, 
    and then splits it into equal train and validation blocks based off
    the specified number of blocks and split percent.
    Returns a list of tuples for the train and validation datasets.
    '''
    # defining the window feature for splitting
    window_spec = Window.partitionBy().orderBy(split_feature)

    # creating a rank column for ordered df
    ranked_df = df.withColumn("rank", percent_rank().over(window_spec))
    
    # creating cross validation percentiles
    train_blocks, val_blocks = cross_val_percentages(blocks, split)

    # Assemble tuples of train and val datasets for cross-validations
    val_train_sets = []
    for train_b, val_b in zip(train_blocks, val_blocks):
        val_train_sets.append((
                                ranked_df.where(f"rank <= {train_b[1]} and rank >= {train_b[0]}").drop('rank')
                                , ranked_df.where(f"rank > {val_b[0]} and rank <= {val_b[1]}").drop('rank')
                                ))
    return val_train_sets

def extract_prob(v):
    '''Convert probability output column to a column with probability of positive.'''
    try:
        return float(v[1])
    except ValueError:
        return None
    
def combining_results(list):
    '''
    Combining a list of dataframes into a single dataframe.
    '''
    results = list[0]
    for result in list[1:]:
        results = results.union(result)
    return results

def decision_tree_model(df_tuple, class_weight, dt_model, verbose=True):
    '''
    Function that builds and fits a decision tree model. 
    Model downsamples dataset for balanced binary label, using class distribution.
    
    Inputs:
     - 'df_tuple': Expects to receive train and validation dfs as a tuple: (train, val). 
     - 'class_weight': percentage of positive binary label.
     - 'dt_model: the initialized model with desired params

    Ouput:
    Returns a dataframe of both validation and train results with 
    columns: ['probability', 'prediction', 'label'].
    '''
    # extracting train & validation from tuple
    train = df_tuple[0]
    val = df_tuple[1]

    #quantifying fraction for downsampling on-time flights
    fraction = class_weight/(1-class_weight)

    if verbose:
            print("Downsampling . . .")

    # downsampling on-time flights
    on_time_train = train[train['label'] == 0].sample(fraction=fraction)
    # temp collection of delayed
    delayed_train = train[train['label'] == 1]
    # recreating downsampled train df
    dwnsmpl_train = on_time_train.union(delayed_train)
    
    if verbose:
        print("Fitting model . . .")
    
    # fit
    dt_fitted = dt_model.fit(dwnsmpl_train)

    # collecting feature importances
    feature_importance = dt_fitted.featureImportances
    
    # get training metrics
    train_metrics = dt_fitted.transform(train)

    # get validation metrics
    val_metrics = dt_fitted.transform(val)

    # retreiving results
    result_cols = ['probability', 'prediction', 'label']
    train_results = train_metrics.select(result_cols)
    validation_results = val_metrics.select(result_cols)

    if verbose:
        print("Results complete.")

    return train_results, validation_results, feature_importance

def TP(prob_pos, label):
    '''Returning an array of 1's for all True Positives by Cutoff'''
    return [ 1 if (prob_pos >= cut_off) and (label == 1)  else 0 for cut_off in CutOffs]
def FP(prob_pos, label):
    '''Returning an array of 1's for all False Positives by Cutoff'''
    return [ 1 if (prob_pos >= cut_off) and (label == 0)  else 0 for cut_off in CutOffs]

def FP_TP_dataframe(df, CutOffs):
    '''
    Function to label a prediction as FP(TP), based on various cutoffs, 
    and map these arrays to df as new columns.
    Expects a dataframe with columns: 'probability', 'label'
    CutOffs to be a list of float percentages.
    '''
    # extracting the delayed flight probabilities from results
    extract_prob_udf = udf(extract_prob, DoubleType())
    df = df.withColumn("prob_pos", extract_prob_udf(col("probability")))

    # Define udfs based on these functions
    # These udfs return arrays of the same length as the cut-off array
    # With 1 if the decision would be TP(FP) at this cut off
    make_TP = udf(TP,  ArrayType(IntegerType()))
    make_FP = udf(FP,  ArrayType(IntegerType()))

    # Generate these arrays in the dataframe returned by prediction
    prediction = df.withColumns({'TP':make_TP(df.prob_pos, df.label), 'FP':make_FP(df.prob_pos, df.label)})

    # Produce a pair-wise sum of these arrays over the entire dataframe, calculate total true positive along the way   
    num_cols = len(CutOffs)
    TP_FP_pd = prediction.agg(array(*[_sum(col("TP")[i]) for i in range(num_cols)]).alias("sumTP"),
                            array(*[_sum(col("FP")[i]) for i in range(num_cols)]).alias("sumFP"),
                            _sum(col("label")).alias("Positives")
                            )\
                            .toPandas()
    # Convert the result into the pd df of precisions and recalls for each cut-off
    results_pd= pd.DataFrame({'Cutoff':CutOffs, 'TP':TP_FP_pd.iloc[0,0], 'FP':TP_FP_pd.iloc[0,1]})
    results_pd['Precision'] = 100*results_pd['TP']/(results_pd['TP'] + results_pd['FP'])
    results_pd['Recall']= 100*results_pd['TP']/TP_FP_pd.iloc[0,2]
    return results_pd

def plot_precision_recall(train_table, val_table):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_table.Recall, train_table.Precision, '-o', label='Train')
    ax.plot(val_table.Recall, val_table.Precision, '-o', label='Validation')
    ax.legend(fontsize=8)
    plt.figtext(0.5, 0.92, "Precision-Recall Performance:\n precision-recall tradeoff at various probability thresholds", ha="center", va="center", fontsize=10)

    ax.set_ylim(10, 50)
    ax.set_xlim(40, 100)
    ax.set_xlabel('Recall (%)', size=10)
    ax.set_ylabel('Precision (%)', size=10)

    # Write cutoff vaulues on the graph
    for index in range(len(train_table.Cutoff)):
        ax.text(train_table.Recall[index]-0.02, 1 + train_table.Precision[index], train_table.Cutoff[index], size=9)
        ax.text(val_table.Recall[index]-0.02, 1 + val_table.Precision[index], val_table.Cutoff[index], size=9)

    # Draw a vertical line to show 80% recall
    ax.axvline(x=80, ymin=0, ymax=60, color='gray', ls = '--')
    ax.text(68, 84, '80% Recall', size=8)

    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-Processing Data

# COMMAND ----------

# Select numeric feature columns
label = 'DEP_DEL15'

# selected features
categorical_features = ['OP_UNIQUE_CARRIER', 
                        'ORIGIN_AIRPORT_ID', 
                        'DEST_AIRPORT_ID', 
                        'origin_type', 
                        'dest_type', 
                        'TAIL_NUM',
                        'OP_CARRIER_FL_NUM'
                        ]
numeric_features = ['origin_6Hr_Precipitation',
                    'dest_6Hr_Precipitation',
                    'origin_12Hr_Visibility',
                    'dest_12Hr_Visibility',
                    'origin_DailySnowfall',
                    'dest_DailySnowfall',
                    'origin_3Hr_DryBulbTemperature',
                    'dest_3Hr_DryBulbTemperature',
                    'origin_DailyDepartureFromNormalAverageTemperature',
                    'dest_DailyDepartureFromNormalAverageTemperature',
                    'origin_3Hr_PressureChange',
                    'dest_3Hr_PressureChange',
                    'origin_12Hr_StationPressure',
                    'dest_12Hr_StationPressure'
                    ]

#features + timestamp + label + splitting column
relev_data = df.select(*categorical_features, *numeric_features, col("sched_depart_date_time_UTC"), col("DEP_DEL15").alias("label"))

#drop na's
relev_data = relev_data.dropna()

# OHE categorical features
result, ohe_features = ohe_features(relev_data, categorical_features)

print("Number of records: ", result.count())

# creating new relevant features list
features = numeric_features + ohe_features

# Combine features into a single vector column using vector assembler
assembler = VectorAssembler(inputCols=features, outputCol="features", handleInvalid="keep")
assembled_df = assembler.transform(result)

print("Review of vectorized features.")
assembled_df.select("features").display(truncate=False)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Create Pure Validation & Cross-Validation Sets

# COMMAND ----------

# craete pure validation and train sets
pure_val_tuples = create_validation_blocks(assembled_df, "sched_depart_date_time_UTC", blocks=1, split=0.8)

# create cross-validations sets
cross_val_tuples = create_validation_blocks(pure_val_tuples[0][0], "sched_depart_date_time_UTC", blocks=3)

# compute label distribution for each cross validation set
class_weights = []
for i, (train, val) in enumerate(cross_val_tuples):
    # developing class weights
    total = train.count()
    delayed = train[train['label'] == 1].count()
    percent = delayed/total
    # print("="*70)
    # print(f"Train Block #{i}")
    # print("Total records: ", total)
    # print("Delayed flights: ", delayed)
    # print("On-Time: ", total-delayed)
    # print("Percent delayed: ", percent)
    class_weights.append(percent)
print("Class Weights: ", class_weights)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Building Model

# COMMAND ----------

# initialize classifier
dt_model = DecisionTreeClassifier(labelCol='label', 
                                    featuresCol='features',
                                    maxDepth = 20, 
                                    maxBins = 60, 
                                    minInstancesPerNode = 100, 
                                    minInfoGain = 0.001
                                    )

# initiating lists for cross-validation results
lst_train_results = []
lst_val_results  = []
lst_feat_import = []
# iterating through each cross-val tuple and running model,
# appending results to list
for i, df_tuple in enumerate(cross_val_tuples):
    print("-"*70)
    print(f"Starting dataset #{i}")
    train_results, validation_results, feature_importances = decision_tree_model(df_tuple, class_weights[i], dt_model)
    lst_train_results.append(train_results)
    lst_val_results.append(validation_results)
    lst_feat_import.append(feature_importances)

# union of all results into a single dataframe
all_train_results = combining_results(lst_train_results)
all_val_results = combining_results(lst_val_results)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Model Evaluation

# COMMAND ----------

# Set decision cut offs
CutOffs = [percentage/100 for percentage in range(20, 61, 10)]
print(CutOffs)

# creating single table of precision & recall
train_table = FP_TP_dataframe(all_train_results, CutOffs)
val_table = FP_TP_dataframe(all_val_results, CutOffs)

# COMMAND ----------

# plotting results
plot_precision_recall(train_table, val_table)
display(train_table)
display(val_table)