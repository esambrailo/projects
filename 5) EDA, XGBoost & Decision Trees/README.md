# mids-207-final-project-summer23-Rueda-Sambrailo-Herr-Liu-Kuehl
Final projeft for MIDS 207

## Contributors
- Erik Sambrailo
- Alberto Lopez Rueda
- Nicole Liu
- Lucy Herr
- Bailey Kuehl

## Dataset 
https://www.kaggle.com/competitions/petfinder-adoption-prediction/rules

## Setting up this repo on your local device
### 1. Go to the dataset website an download it to your local device. Rename the file "data" if it is not named that already.
Unfortunately, the files are gigantic, and GitHub doesn't allow you to upload files > 25MB.
Naming the file "data" will ensure that you can run the files in step 2 below without changing your path. If you prefer, you can change your path to access the data and name it whatever you like.

### 2. Clone this repo to your local device
```git clone https://github.com/UC-Berkeley-I-School/mids-207-final-project-summer23-Rueda-Sambrailo-Herr-Liu-Kuehl.git```

### 3. Move the "data" file containing the PetFinder dataset into the cloned repo.


### 4. Run the files in successive order.
1. 1_Parsing_and_Mering_AllData -- this will take all of the raw data from the data folder, parse it into columns, and merge it on PetID. It will also split the data into test and train. By using "random_state = 1", all executions of this notebook will result in the same test/train split.

2. 2_DataExploration_Charts -- Optionally, you can run this file to see data distributions, charts, etc. that were used to produce the cleaned features created in step 2.

3. 3_Features_DataCleaning -- this will perform feature engineering (data cleaning, transformations) on all data.

4. 4_ModelTraining -- this will train 3 different models, including: baseline, FF neural network, and transformers.

## Pulling most recent data
You should do all work in your individual branch (once you have pulled from main). Here are the steps you should take:

1. Create your own branch: git branch <branch_name>
2. Switch to your branch: git checkout <branch_name>
3. Get any updates from the main branch into your branch:
   
      ```git fetch origin       # downloads new data from a remote repository - doesn't integrate any of this new data into your working branch```    
      ```git pull origin main   # pull changes from the origin remote main branch and merge them to the local branch: branch_name```  
5. You should now have any changes from main. You are still in your working branch.
6. Proceed making changes, followed by    
     ```git status```   
     ```git add <changed file name>```  
     ```git commit -m "your message"```     
     ```git push origin <branch_name>```   
7. If you want your changes to be reflected in the main branch, you should now go into GitHub in the browser, find your commit, and start Pull Request.
   

    
