# World Happiness Data 2022

## Project Objective

Exploratory Data Analysis on the World Happiness Data of 146 countries, to explore the correlation between Per Capita GDP, Freedom to Make Life Choices, Perceptions of Corruption, Life Expectancy, Social Support, Generosity.

## Data Used

<a href = "https://www.kaggle.com/datasets/mathurinache/world-happiness-report"> The World Happiness Report 2022 </a> is a publication of the Sustainable Development Solutions Network, under the initiative of the United Nations, using data Primarily from the Gallup World Poll. 2022 is the 10th anniversary of the report, which has been published since 2013.

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

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/View%20of%20the%20Data.png)
    
### Step 4:

Remove unnecessary columns and rename the rest

    # Removing unncessary columns

    data_columns = ['Country','Regional indicator','Happiness score','Explained by: GDP per capita','Explained by: Social support','Explained by: Healthy life expectancy','Explained by: Freedom to make life choices','Explained by: Generosity','Explained by: Perceptions of corruption']
    data = data[data_columns].copy()

    data.head()
    
<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/After%20Removing%20Columns.png)

    # Renaming columns

    happy_df = {'Country':'country','Regional indicator':'regional_indicator','Happiness score':'happiness_score','Explained by: GDP per capita':'gdp_per_capita','Explained by: Social support':'social_support','Explained by: Healthy life expectancy':'healthy_life_expectancy','Explained by: Freedom to make life choices':'freedom_to_make_life_choices','Explained by: Generosity':'generosity','Explained by: Perceptions of corruption':'perceptions_of_corruption'}
    data = data.rename(columns = happy_df)
    data.head()

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/After%20Renaming%20Columns.png)

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

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Null%20Values.png)

Remove null values

    # Removing null values

    data = data[data.notnull().all(1)]
    data
    
<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Removing%20Null%20Values.png)

## Data Analysis Process

### STEP 1: Total Number of Countries in each Region
 
For this analysis, the countries were split into 10 different regions:

    # Total Countries by Region

    total_country = data.groupby('regional_indicator')[['country']].count()
    total_country

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Total%20Countries%20by%20Region.png)

### STEP 2: Correlation Map between Happiness and Six Different Factors Affecting Happiness

    # Correlation Map

    cor = data.corr(method = 'pearson')
    f, ax = plt.subplots(figsize = (10,5))
    sns.heatmap(cor, mask = np.zeros_like(cor,dtype = np.bool),cmap = 'Reds',square = True,ax = ax)
    
<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Correlation%20Map.png)

### STEP 3: Happiness Score and Social Support

    # Social Support and Happiness

    plt.rcParams['figure.figsize'] = (15,7)
    plt.title('Plot between Happiness and Social Support')
    sns.scatterplot(x = data.social_support,y = data.happiness_score,hue = data.regional_indicator,s = 200)

    plt.legend(loc = 'upper left',fontsize = '12')
    plt.xlabel('Social Support')
    plt.ylabel('Happiness Score')

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Plot%20between%20Happiness%20and%20Social%20Support.png)

### STEP 4: Happiness Score and GDP Per Capita

    # Plot between Happiness and GDP Per Capita

    plt.rcParams['figure.figsize'] = (15,7)
    plt.title('Plot between Happiness and GDP Per Capita')
    sns.scatterplot(x = data.happiness_score,y = data.gdp_per_capita,hue = data.regional_indicator,s = 200)

    plt.legend(loc = 'lower right',fontsize = '10')
    plt.xlabel('Happiness Score')
    plt.ylabel('GDP Per Capita')

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Plot%20Between%20Happiness%20and%20GDP%20Per%20Capita.png)

### STEP 5: Happiness Score and Healthy Life Expectancy

    # Healthy Life Expectancy and Happiness

    plt.rcParams['figure.figsize'] = (15,7)
    plt.title('Plot between Happiness and Healthy Life Expectancy')
    sns.scatterplot(x = data.happiness_score,y = data.healthy_life_expectancy,hue = data.regional_indicator,s = 200)

    plt.legend(loc = 'lower right',fontsize = 14)
    plt.xlabel('Happiness Score')
    plt.ylabel('Healthy Life Expectancy')

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Plot%20between%20Happiness%20and%20Healthy%20Life%20Expectancy.png)

### STEP 6: Happiness Score and Freedom to Make Life Choices

    # Freedom to Make Life Choices and Happiness

    plt.rcParams['figure.figsize'] = (15,7)
    plt.title('Plot between Happiness and Freedom to Make Life Choices')
    sns.scatterplot(x = data.freedom_to_make_life_choices,y = data.happiness_score,hue = data.regional_indicator,s = 200)

    plt.legend(loc = 'upper left',fontsize = '12')
    plt.xlabel('Freedom to Make Life Choices')
    plt.ylabel('Happiness Score')

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Plot%20between%20Happiness%20and%20Freedom%20to%20Make%20Life%20Choices.png)

### STEP 7: Happiness Score and Perception of Corruption

    # Corruption and Happiness

    plt.rcParams['figure.figsize'] = (15,7)
    plt.title('Plot between Happiness and Corruption')
    sns.scatterplot(x = data.happiness_score,y = data.perceptions_of_corruption,hue = data.regional_indicator,s = 200)

    plt.legend(loc = 'upper left',fontsize = 14)
    plt.xlabel('Happiness Score')
    plt.ylabel('Corruption Index')
    
<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Plot%20Between%20Happiness%20and%20Corruption.png)

### STEP 8: Happiness Score and Generosity

    # Generosity and Happiness

    plt.rcParams['figure.figsize'] = (15,7)
    plt.title('Plot between Happiness and Generosity')
    sns.scatterplot(x = data.generosity,y = data.happiness_score,hue = data.regional_indicator,s = 200)

    plt.legend(loc = 'lower right',fontsize = '12')
    plt.xlabel('Generosity')
    plt.ylabel('Happiness Score')

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Plot%20between%20Happiness%20and%20Generosity.png)

### STEP 9: Factors Contributing to Happiness - Regions



### STEP 10: Factors Contributing to Happiness - Top 10 Countries in the World

    # Top 10 Happiest Countries

        top_10 = data.head(10)
        top_10
    
<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Top%2010%20Happiest%20Countries.png)

### STEP 11: Factors Contributing to Happiness - Bottom 10 Countries in the World

    # Bottom 10 Unhappiest Countries

    bottom_10 = data.tail(10)
    bottom_10

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Top%2010%20Unhappiest%20Countries.png)

## Insights:

1. GDP Per Capita, Social Support and Healthy Life Expectancy had heavy influences on the Happiness score while Freedom to Make Life Choices had a moderate influence, Perceptions of Corruption had very little influence and Generosity had almost no influence at all
2. 8 out of the Top 10 nations in terms of Happiness score are in Western Europe while 8 out of the Bottom 10 nations in terms of Happiness score are in Sub-Saharan Africa
3. 5 out of the 10 Happiest Nations are in the Top 10 in terms of GDP Per Capita while only 3 out of the 10 Unhappiest Nations were in the Bottom 10 in terms of GDP Per Capita
4. 7 out of the 10 Happiest Nations were also among the Top 10 best in terms of Perception of Corruption while Afghanistan and Lesotho were the only two nations in the Bottom 10 in both Happiness and Perception of Corruption.
5. 5 out of the 10 Unhappiest countries figured in the Bottom 10 in terms of Healthy Life Expectancy as well while Israel and Switzerland were in the only nations in the Top 10 in both Happiness and Healthy Life Expectancy
6. 6 out of the 10 Happiest Countires were in the Top 10 in terms of Freedom to Make Life Choices while Lebanon and Afghanistan were not only the only nations in the Bottom 10 in terms of Both Happiness and Freedom to Make Life Choices, but also the Bottom 2 in both
7. 5 out of the 10 Happiest Countries were in the Top 10 in terms of Social Support while 4 out of the 10 Unhappiest Countries were in the Bottom 10 in terms of Social Support
8. Netherlands is the only nation that was in the Top 10 in terms of both Happiness and Generosity Botswana was the only nation to be in the Bottom 10 in both these categories 

## Conclusions:

1. A country should increase its resources, provide means for a longer living and offer good societal support from its people to increase its Happiness Score.
2. The ranking of a country in each indicator individually doesn't necessarily affect its ranking in the Happiness index to a great extent, except for the Corruption and Freedom to Make Life Choices ranking. Even in these indicators, the Bottom 10 differs almost entirely from that of the overall Happiness Score
