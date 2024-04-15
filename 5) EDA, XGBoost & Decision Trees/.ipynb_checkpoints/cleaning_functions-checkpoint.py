import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from mlxtend.plotting import heatmap
from wordcloud import WordCloud
import openpyxl

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import category_encoders as ce

import os

#------------------------------------------------------------------------    
# 1. ES - CLEANING FUNCTION
# erik 
def clean_es_features(train, colors): 
    #Type - transforming all 2's to 0's
    train['Type'] = train['Type'].replace(2, 0) 
    #has_Video - transforming all non 0's to 1's
    train['has_Video'] = (train['VideoAmt'] != 0).astype(int)
    #has_Photo - transforming all non 0's to 1's
    train['has_Photo'] = (train['PhotoAmt'] != 0).astype(int)
    #MaturitySize - replacing all 0's with -1's
    train['MaturitySize'] = train['MaturitySize'].replace(0, -1)
    #Maturity_isSpecified
    train['Maturity_isSpecified'] = (train['MaturitySize'] != 0).astype(int)
    #FurLength - replacing all 0's with -1's
    train['FurLength'] = train['FurLength'].replace(0, -1)
    #FurLength_isSpecified
    train['FurLength_isSpecified'] = (train['FurLength'] != 0).astype(int)
    #isMale - transform to binary
    train['isMale'] = train['Gender'].apply(lambda x: 1 if x == 1 or x == 3 else 0) 
    #isFemale - transform to binary
    train['isFemale'] = train['Gender'].apply(lambda x: 1 if x == 2 or x == 3 else 0) 
    #{Color} - OHE for presence of each color
    for color_num, color in zip(colors['ColorID'], colors['ColorName']):
        train[color] = train[['Color1', 'Color2', 'Color3']].apply(lambda row: 1 if color_num in row.values else 0, axis=1) 
    #ColorCount
    color_columns = colors['ColorName'].tolist()
    train['ColorCount'] = train[color_columns].sum(axis=1)
    return train

# Define function to OHE
def OHE_vars(df, cols):     
    df_selected = df[cols]
    # Creating an instance of the OneHotEncoder
    encoder = OneHotEncoder(categories='auto', sparse_output=False)
    # Encoding the selected columns
    encoded_columns = encoder.fit_transform(df_selected)
    # Creating new column names based on the categories
    new_column_names = encoder.get_feature_names_out(cols)
    # Creating a new DataFrame with the encoded columns
    df_encoded = pd.DataFrame(encoded_columns, columns=new_column_names)
    # Combining the encoded DataFrame with the original DataFrame
    df = pd.concat([df, df_encoded], axis=1)
    #df = pd.concat([df.drop(cols, axis=1), df_encoded], axis=1)
    return df

# Binary Encoding
def binary_encoding(df, target):
    # Check if column_name exists in the DataFrame
    if target not in df.columns:
        raise ValueError(f"Column '{target}' does not exist in the DataFrame.")
    # Encode
    temp_encoder = ce.BinaryEncoder(cols=[target])
    df = temp_encoder.fit_transform(df)
    return df 
   # Relabel numeric values to [0,2] instead of [1,3]

# Relabel numeric values to [0,2] instead of [1,3]
def relabel_col(df, target, to_range, from_range):
    # Check if column_name exists in the DataFrame
    if target not in df.columns:
        raise ValueError(f"Column '{target}' does not exist in the DataFrame.")       
    # Create dict and apply function
    dict_lab = {i: j for i, j in zip(to_range, from_range)}
    df[target] = df[target].map(dict_lab)
    return df

# Normalizing function 
def normalize_var(df, target):
    # Create an instance of StandardScaler
    scaler = StandardScaler()
    # Select column and reshape it to a 2-dimensional array
    target_values = df[target].values.reshape(-1, 1)
    # Fit and transform the 'Fee' values using StandardScaler
    target_scaled = scaler.fit_transform(target_values)
    # Replace the 'Fee' column in the DataFrame with the scaled values
    df[target] = target_scaled
    return df


    
    