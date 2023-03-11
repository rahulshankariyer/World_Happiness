# World Happiness Data 2022

## Project Objective

Exploratory Data Analysis on the World Happiness Data of 146 countries, based on Per Capita GDP, Freedom to Make Life Choices, Perceptions of Corruption, Life Expectancy, etc.

## Data Used

<a href = "https://www.kaggle.com/datasets/mathurinache/world-happiness-report"> The World Happiness Report 2022 </a> is a publication of the Sustainable Development Solutions Network, under the iniative of the United Nations, using data Primarily from the Gallup World Poll. 2022 is the 10th anniversary of the report, which has been published since 2013.

## Tools Used

Python Libraries - Pandas, Numpy, Seaborn and Matplotlib

## Data Cleaning Steps

### Step 1:

Import all necessary Python libraries (in Jupyter Notebook)

    # Importing all libraries

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    %matplotlib inline
    
### Step 2:

Format the graphs

    # Formatting the graphs

    sns.set_style('darkgrid')
    plt.rcParams['font.size'] = 15
    plt.rcParams['figure.figsize'] = (10,7)
    plt.rcParams['figure.facecolor'] = '#17becf'
    
### Step 3:

Extract the data from the CSV file and view the data

    # Extracting the data from CSV file

    data = pd.read_csv('C:\\Users\\rahulshankariyer\\Documents\\2022.csv')
    
    # Viewing the data

    data.head()
    
<b> Output: </b>


    
### Step 4:

Remove unnecessary columns and rename the rest

    # Removing unncessary columns

    data_columns = ['Country','Regional indicator','Happiness score','Explained by: GDP per capita','Explained by: Social support','Explained by: Healthy life expectancy','Explained by: Freedom to make life choices','Explained by: Generosity','Explained by: Perceptions of corruption']
    data = data[data_columns].copy()

    data.head()
    
<b> Output: </b>



    # Renaming columns

    happy_df = {'Country':'country','Regional indicator':'regional_indicator','Happiness score':'happiness_score','Explained by: GDP per capita':'gdp_per_capita','Explained by: Social support':'social_support','Explained by: Healthy life expectancy':'healthy_life_expectancy','Explained by: Freedom to make life choices':'freedom_to_make_life_choices','Explained by: Generosity':'generosity','Explained by: Perceptions of corruption':'perceptions_of_corruption'}
    data = data.rename(columns = happy_df)
    data.head()

<b> Output: </b>



### Step 5:

Check for null values, investigate if any and remove them

    # Checking for null values

    data.isnull().sum()
    
<b> Output: </b>

country                         0
regional_indicator              1
happiness_score                 1
gdp_per_capita                  1
social_support                  1
healthy_life_expectancy         1
freedom_to_make_life_choices    1
generosity                      1
perceptions_of_corruption       1
dtype: int64  

Investigate null values (if any)

    # Investigating null values

    data[data.isnull().any(axis=1)]
    
<b> Output: </b>



Remove null values

    # Removing null values

    data = data[data.notnull().all(1)]
    data
    
<b> Output: </b>



## Data Analysis Process
 
For many parts of this analysis, the countries were split into 10 different regions:

1. North America and ANZ
2. Latin America and Carribean
3. Western Europe
4. Central and Eastern Europe
5. Commonwealth of Independent States
6. Middle East and North Africa
7. Sub-Saharan Africa
8. South Asia
9. Southeast Asia
10. East Asia

Lets take a look into the total number of countries in each region

    #Total Countries by Region

    total_country = data.groupby('regional_indicator')[['country']].count()
    total_country

<b> Output: </b>



Let's look at which countries are at the top of the happiness charts and which ones are at the bottom

    # Top 10 Happiest Countries

    top_10 = data.head(10)
    top_10
    
<b> Output: </b>



    # Top 10 Unhappiest Countries

    bottom_10 = data.tail(10)
    bottom_10

<b> Output: </b>

Let's investigate the relationship between the Happiness Score and GDP Per Capita of the countries, with a specific focus on region



## Insights:

1. 8 out of the Top 10 nations in terms of Happiness score are in Western Europe while 8 out of the Bottom 10 nations in terms of Happiness score are in Sub-Saharan Africa
2. 

