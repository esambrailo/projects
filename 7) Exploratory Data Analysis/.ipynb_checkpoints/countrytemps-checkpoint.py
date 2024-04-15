import numpy as np
import pandas as pd

def country_anomoly(country_file):
    '''
    function that extracts and wrangles the anomoly temperature 
    data into DataFrame from country.txt file.
    '''
    columns = ['Year', 'Month', 'Monthly_Anomaly', 'Monthly_Unc', 'Annual_Anomaly', 'Annual_Unc',
               'Five_year_Anomaly', 'Five_year_Unc.', 'Ten_year_Anomaly', 'Ten_year_Unc.',
               'Twenty_year_Anomaly','Twenty_year_Unc.']
    country = pd.read_table(f'./country_data/{country_file}') #read in the text
    country = country[~country.iloc[:,0].str.contains('%')] #filter out any column starting with a '%'
    country = country.iloc[:,0].str.split(expand = True) #split the column and expand out
    country = country.reset_index(drop = True) #reset the index and drop the old one
    country.columns = columns #rename the columns to equal the columns list
    
    #assign dtypes for the columns
    for column in country.columns[:2]:
        country[column] = country[column].astype('int')
    for column in country.columns[2:]:
        country[column] = country[column].astype('float')
    
    country = country[~country.Monthly_Anomaly.isnull()]
    return country

def country_baselines(country_file):
    '''
    function that extracts and wrangles the baseline temperature
    data into DataFrame from country.txt file.
    '''
    columns = ['Month', 'Baseline_Temp', 'Unc']
    country = pd.read_table(f'./country_data/{country_file}') #read in the text
    country = country.iloc[52:55,] #specifically looking for rows that contain baseline data
    country = country.iloc[:,0].str.split(expand = True).T #split the column and expand out
    country.columns = columns #rename the columns to equal the columns list
    country['Unc'] = country.iloc[:,2].shift(-1) #shifting uneven column to align
    country.drop(index = [0,13], inplace = True) #dropping extra un-needed rows
    country = country.reset_index(drop = True) #reset the index and drop the old one
    return country