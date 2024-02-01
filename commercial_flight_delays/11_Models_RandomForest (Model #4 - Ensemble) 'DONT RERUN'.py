# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Random Forest Ensemble
# MAGIC Notebook for combining three Random Forest models of varying timeframes into a single ensemble.
# MAGIC
# MAGIC Three random forests were trained:
# MAGIC  - (1) at full train timeframe
# MAGIC  - (1) at latest 1/2 of train timeframe
# MAGIC  - (1) at latest 1/4 of train timeframe
# MAGIC
# MAGIC Using this notebook to explore the best way to combine the three random forest models. 

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
from pyspark.sql.functions import col, percent_rank, when, udf, array, sum as _sum, monotonically_increasing_id, row_number, lit
from pyspark.ml.functions import vector_to_array
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import Row
from pyspark.sql.window import Window
from pyspark.sql.types import ArrayType, IntegerType, DoubleType, StructType, StructField, StringType, ByteType, FloatType
from pyspark.sql.functions import udf
from pyspark.ml.linalg import DenseVector

# RF model
from pyspark.ml.classification import RandomForestClassifier


# COMMAND ----------

# MAGIC %md 
# MAGIC ## Datasets & Model Parameters

# COMMAND ----------

# read in the three different model results
team_blob_url = funcs.blob_connect()
directories = {
            'cross_train': 'ES/RF/Model3_grid_ctrain/',
            'cross_val': 'ES/RF/Model3_grid_cval/',
            'long_train': 'ES/RF/Model4_long_train',
            'long_val': 'ES/RF/Model4_long_val',
            'mid_train': 'ES/RF/Model4_mid_train',
            'mid_val': 'ES/RF/Model4_mid_val',
            'short_train': 'ES/RF/Model4_short_train',
            'short_val': 'ES/RF/Model4_short_val'
            }

results = {}
for name, path in directories.items():
    results[name] = spark.read.parquet(f"{team_blob_url}/{path}").cache()

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

        # # dropping unnecessary columns
        # df = df.drop(*[feature, f"{feature}_index"])
        # adding new feature to list
        ohe_features.append(f"{feature}_encoded")
    return df, ohe_features

def readable_feature_list(df, categorical_features, numerical_features):

    # initiating list and index counter
    lbl_lst = []
    lbl_idx = 0
    # iterate through each categorical feature
    for feature in categorical_features:
        # creating an ordered list of all indexed values for a categorical feature
        lbls = [c.metadata["ml_attr"]["vals"] for c in df.schema.fields if c.name == (f"{feature}_index")][0]
        for lbl in lbls[:-1]:
            lbl_lst.append((lbl_idx, f'{feature} = {lbl}'))
            lbl_idx +=1

    # Add numerical features to this list features
    for ft in numerical_features:
        lbl_lst.append((lbl_idx, ft))
        lbl_idx +=1
    return lbl_lst
    

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

def random_forest_model(df_tuple, class_weight, rf_model, verbose=True):
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
    rf_fitted = rf_model.fit(dwnsmpl_train)

    # collecting feature importances
    feature_importance = rf_fitted.featureImportances
    
    # get training metrics
    train_metrics = rf_fitted.transform(train)

    # get validation metrics
    val_metrics = rf_fitted.transform(val)

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

def plot_precision_recall(train_table, val_table, xlims =(0,100), ylims=(0,100)):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_table.Recall, train_table.Precision, '-o', label='Train')
    ax.plot(val_table.Recall, val_table.Precision, '-o', label='Validation')
    ax.legend(fontsize=8)
    plt.figtext(0.5, 0.92, "Precision-Recall Performance:\n precision-recall tradeoff at various probability thresholds", ha="center", va="center", fontsize=10)

    ax.set_ylim(ylims[0], ylims[1])
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_xlabel('Recall (%)', size=10)
    ax.set_ylabel('Precision (%)', size=10)

    # Write cutoff vaulues on the graph
    for index in range(len(train_table.Cutoff)):
        ax.text(train_table.Recall[index]-0.02, 1 + train_table.Precision[index], train_table.Cutoff[index], size=9)
        ax.text(val_table.Recall[index]-0.02, 1 + val_table.Precision[index], val_table.Cutoff[index], size=9)

    # Draw a vertical line to show 80% recall
    ax.axvline(x=80, ymin=0, ymax=60, color='gray', ls = '--')
    # ax.text(68, 84, '80% Recall', size=8)

    plt.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Model Evaluation

# COMMAND ----------

# Set decision cut offs
CutOffs = [percentage/100 for percentage in range(45, 51, 1)]
print(CutOffs)

# COMMAND ----------

# creating single table of precision & recall
c_train_table = FP_TP_dataframe(results['cross_train'], CutOffs)
c_val_table = FP_TP_dataframe(results['cross_val'], CutOffs)

# plotting results
plot_precision_recall(c_train_table, c_val_table,  xlims =(55,100), ylims=(15,40))
display(c_train_table)
display(c_val_table)

# COMMAND ----------

# creating single table of precision & recall
long_train_table = FP_TP_dataframe(results['long_train'], CutOffs)
long_val_table = FP_TP_dataframe(results['long_val'], CutOffs)

# plotting results
plot_precision_recall(long_train_table, long_val_table,  xlims =(55,100), ylims=(15,40))
display(long_train_table)
display(long_val_table)

# COMMAND ----------

# creating single table of precision & recall
mid_train_table = FP_TP_dataframe(results['mid_train'], CutOffs)
mid_val_table = FP_TP_dataframe(results['mid_val'], CutOffs)

# plotting results
plot_precision_recall(mid_train_table, mid_val_table,  xlims =(55,100), ylims=(15,40))
display(mid_train_table)
display(mid_val_table)

# COMMAND ----------

# creating single table of precision & recall
short_train_table = FP_TP_dataframe(results['short_train'], CutOffs)
short_val_table = FP_TP_dataframe(results['short_val'], CutOffs)

# plotting results
plot_precision_recall(short_train_table, short_val_table,  xlims =(55,100), ylims=(15,40))
display(short_train_table)
display(short_val_table)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Creating Ensemble of 3 Models

# COMMAND ----------

display(results['long_val'])

# COMMAND ----------

# checking that each validation set contains the same exact number of records. 
for val in results.keys():
    # only looking at validation records
    if 'val' in val:
        # adding a sequential id to each df for joining
        windowSpec = Window.orderBy(monotonically_increasing_id())
        results[val] = results[val].withColumn("id", row_number().over(windowSpec))
        print(f"Record count for {val}: {results[val].count()}")

# COMMAND ----------

display(results['short_val'])

# COMMAND ----------

# joining two validation dfs together
def join_results(df_1, df_2, df_3, name_1, name_2, name_3, cols_1, cols_2, cols_3):
    df_1 = df_1.select(cols_1)
    df_2 = df_2.select(cols_2)
    df_3 = df_3.select(cols_3)
    results = df_1.alias("df_1").join(df_2.alias("df_2"), ['sched_depart_date_time_UTC', 'TAIL_NUM']) \
                                .join(df_3.alias("df_3"), ['sched_depart_date_time_UTC', 'TAIL_NUM']) \
                    .withColumn(name_1, vector_to_array(f"df_1.{cols_1[-1]}")) \
                    .withColumn(name_2, vector_to_array(f"df_2.{cols_2[-1]}")) \
                    .withColumn(name_3, vector_to_array(f"df_3.{cols_3[-1]}")) \
                    .drop(cols_1[-1])
    return results

cols_1 = ['sched_depart_date_time_UTC', 'TAIL_NUM', 'label', 'probability']
cols_2 = ['sched_depart_date_time_UTC', 'TAIL_NUM', 'probability']
cols_3 = ['sched_depart_date_time_UTC', 'TAIL_NUM', 'probability']
joined_validation_results = join_results(results['long_val'], results['mid_val'], results['short_val'], "long", "mid", "short", cols_1, cols_2, cols_3)

display(joined_validation_results)

# COMMAND ----------

# defining function for calculating averages
# Define a UDF to calculate the average of two arrays
@udf(ArrayType(FloatType()))
def calculate_avg(prob1, prob2, prob3):
    avg_values = [(a + b + c ) / 3 for a, b, c in zip(prob1, prob2, prob3)]
    return avg_values

# Define the UDF
@udf(returnType=IntegerType())
def count_values_greater_than_05(*arrays):
    count = 0
    for array in arrays:
        if array[1] > 0.5:
            count += 1
    return count

# Create a new column with the averages
avg_results = joined_validation_results.withColumn("probability", calculate_avg(col("long"), col("mid"), col("short"))) \
                                        .withColumn("pred_count", count_values_greater_than_05(col("long"), col("mid"), col("short")))

display(avg_results)

# saving results to parquet
funcs.write_parquet_to_blob(avg_results, 'ES/RF/Model4_ensemble_val')

# COMMAND ----------

# creating single table of precision & recall
ensemble_val_table = FP_TP_dataframe(avg_results, CutOffs)
long_val_table = FP_TP_dataframe(results['long_val'], CutOffs)

# plotting results
plot_precision_recall(long_val_table, ensemble_val_table,  xlims =(55,100), ylims=(15,40))
display(long_val_table)
display(ensemble_val_table)

# COMMAND ----------

