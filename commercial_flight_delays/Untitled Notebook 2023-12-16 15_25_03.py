# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC The top performing model (only slightly better than the logistic regression) is the Random Forest. This is likely because of its ability to capture non-linear patterns with little compute power. Since our target metric was to maximize precision while holding recall at 80%, we had to get creative with how to accomplish this with a random forest model. Instead of extracting the models predictions, we instead extracted the ratios of on-time vs. delayed flights for each leaf node. We translated this ratio of labels as the predicted probability of delay. A record was then given the 'probability' of the node for which it fell into. For hyper-parameter tuning we tested a grid search during our cross-validation stages of modeling. Below is a table showing the parameters of the grid search, as well as a table for the best performing parameter combinations for each fold. 
# MAGIC
# MAGIC **Hyperparmeters for Grid Search**
# MAGIC
# MAGIC | Hyper Parameter | List of Values |
# MAGIC |-----------------| -------------- |
# MAGIC | 'maxDepth'      |  [5, 10, 15]   |
# MAGIC | 'numTrees'      |  [50, 75, 100] |
# MAGIC | 'minInfoGain'   |  [0.01, 0.001, 0.0001] |
# MAGIC
# MAGIC **Best Performing Combinations**
# MAGIC
# MAGIC | Fold | Parameters |
# MAGIC |-----| -------------- |
# MAGIC | 1   |  'maxDepth': 15 <br> 'numTrees': 50 <br> 'minInfoGain': 0.0001 |
# MAGIC | 2   | 'maxDepth': 15 <br> 'numTrees': 100 <br> 'minInfoGain': 0.0001 |
# MAGIC | 3   |  'maxDepth': 15 <br> 'numTrees': 50 <br> 'minInfoGain': 0.0001 |
# MAGIC
# MAGIC
# MAGIC We only ran the grid search once, and for a couple different reasons. First, due to the expensive nature of grid searches, we wanted to limit our usage unless it was providing significant value. Second, time contrainsts limited our options to pre-made grid search modules available for pyspark. This created two issues for us: 1) The grid search we found available requires accompanying validation folds, and those folds do not meet time-series requirements. 2) The uniqueness of our selected metric (precision at held 80% recall) is not one that is readily available as an evaluator of the grid search.  For the grid search we settled with AUC as the evaluator.  For these reasons we kept the results of the grid search in mind, but carried forward with more hueristic manual tuning of the RF parameters.  With more time, we would have opted to develop a custom grid search that addresses these two concerns.  
# MAGIC
# MAGIC The RF model only performing slighly better than LR, reinforces the notion that the data set lacks features with predictive power. However, on a someone divergent note, had our objective been more towards holding a high standard for precision while attempting to maximize recall, this model may have shown useful. 
# MAGIC
# MAGIC grid search

# COMMAND ----------

Param Map 9 - Avg Metric: 0.7534
  numTrees: 50
  maxDepth: 15
  minInfoGain: 0.0001

Param Map 27 - Avg Metric: 0.7553
  numTrees: 100
  maxDepth: 15
  minInfoGain: 0.0001

Param Map 9 - Avg Metric: 0.7567
  numTrees: 50
  maxDepth: 15
  minInfoGain: 0.0001
