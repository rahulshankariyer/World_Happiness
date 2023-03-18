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
    plt.rcParams['figure.facecolor'] = '#FAEBD7'
    
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

    # Removing unnecessary columns

    data_columns = ['Country','Regional indicator','Happiness score','Dystopia (1#83) + residual','Explained by: GDP per capita','Explained by: Social support','Explained by: Healthy life expectancy','Explained by: Freedom to make life choices','Explained by: Generosity','Explained by: Perceptions of corruption']

    data = data[data_columns].copy()
    data.head()
    
<b> Output: </b>

![alt text]()

    # Renaming columns

    happy_df = {'Country':'country','Regional indicator':'regional_indicator','Happiness score':'happiness_score','Dystopia (1#83) + residual':'dystopia','Explained by: GDP per capita':'gdp_per_capita','Explained by: Social support':'social_support','Explained by: Healthy life expectancy':'healthy_life_expectancy','Explained by: Freedom to make life choices':'freedom_to_make_life_choices','Explained by: Generosity':'generosity','Explained by: Perceptions of corruption':'perceptions_of_corruption'}
    data = data.rename(columns = happy_df)
    data.head()

<b> Output: </b>

![alt text]()

### Step 5:

Check for null values, investigate if any and remove them

    # Checking for null values

    data.isnull().sum()
    
<b> Output: </b>

    country                         0
    regional_indicator              1
    happiness_score                 1
    dystopia                        1
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

![alt text]()

Remove null values

    # Removing null values

    data = data[data.notnull().all(1)]
    data
    
<b> Output: </b>

![alt text]()

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
    sns.heatmap(cor, mask = np.zeros_like(cor,dtype = np.bool),cmap = 'Reds',square = True,ax = ax,annot = True)
    
<b> Output: </b>

![alt text]()

### STEP 3: Happiness Score and Social Support

    # Social Support and Happiness

    plt.rcParams['figure.figsize'] = (15,7)
    plt.rcParams['figure.facecolor'] = '#17becf'
    plt.title('Plot between Happiness and Social Support')
    sns.scatterplot(x = data.social_support,y = data.happiness_score,hue = data.regional_indicator,s = 200)

    plt.legend(loc = 'upper left',fontsize = '12')
    plt.xlabel('Social Support')
    plt.ylabel('Happiness Score')

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Plot%20between%20Happiness%20and%20Social%20Support.png)

    # Social Support by Regions

    social_support = data.groupby('regional_indicator')[['social_support']].mean()
    social_support
    
<b> Output: </b>

![alt text]()

    social_support = social_support.sort_values('social_support',ascending = False)

    plt.rcParams['figure.figsize'] = (12,8)
    plt.title('Social Support in Various Regions')
    plt.xlabel('Regions',fontsize = 15)
    plt.ylabel('Social Support',fontsize = 15)
    plt.xticks(rotation = 30,ha = 'right')
    plt.bar(social_support.index,social_support.social_support,color = ['C3','C6','C8','C4','C1','C2','C9','C5','C7','C0'])

<b> Output: </b>

![alt text]()

### STEP 4: Happiness Score and GDP Per Capita

    # Plot between Happiness and GDP Per Capita

    plt.rcParams['figure.figsize'] = (15,7)
    plt.title('Plot between Happiness and GDP Per Capita')
    sns.scatterplot(x = data.gdp_per_capita,y = data.happiness_score,hue = data.regional_indicator,s = 200)

    plt.legend(loc = 'lower right',fontsize = '10')
    plt.xlabel('GDP Per Capita')
    plt.ylabel('Happiness Score')

<b> Output: </b>

![alt text]()

    # GDP Per Capita by Regions

    gdp_per_capita = data.groupby('regional_indicator')[['gdp_per_capita']].mean()
    gdp_per_capita

<b> Output: </b>

![alt text]()

    gdp_per_capita = gdp_per_capita.sort_values('gdp_per_capita',ascending = False)

    plt.rcParams['figure.figsize'] = (12,8)
    plt.title('GDP Per Capita in Various Regions')
    plt.xlabel('Regions',fontsize = 15)
    plt.ylabel('GDP Per Capita',fontsize = 15)
    plt.xticks(rotation = 30,ha = 'right')
    plt.bar(gdp_per_capita.index,gdp_per_capita.gdp_per_capita,color = ['C3','C6','C8','C4','C1','C2','C9','C5','C7','C0'])

<b> Output: </b>

![alt text]()

### STEP 5: Happiness Score and Healthy Life Expectancy

    # Healthy Life Expectancy and Happiness

    plt.rcParams['figure.figsize'] = (15,7)
    plt.title('Plot between Happiness and Healthy Life Expectancy')
    sns.scatterplot(x = data.healthy_life_expectancy,y = data.happiness_score,hue = data.regional_indicator,s = 200)

    plt.legend(loc = 'lower right',fontsize = 14)
    plt.xlabel('Healthy Life Expectancy')
    plt.ylabel('Happiness Score')

<b> Output: </b>

![alt text]()

    # Healthy Life Expectancy by Regions

    healthy_life_expectancy = data.groupby('regional_indicator')[['healthy_life_expectancy']].mean()
    healthy_life_expectancy

<b> Output: </b>

![alt text]()

    healthy_life_expectancy = healthy_life_expectancy.sort_values('healthy_life_expectancy',ascending = False)

    plt.rcParams['figure.figsize'] = (12,8)
    plt.title('Healthy Life Expectancy in Various Regions')
    plt.xlabel('Regions',fontsize = 15)
    plt.ylabel('Healthy Life Expectancy',fontsize = 15)
    plt.xticks(rotation = 30,ha = 'right')
    plt.bar(healthy_life_expectancy.index,healthy_life_expectancy.healthy_life_expectancy,color = ['C3','C6','C8','C4','C1','C2','C9','C5','C7','C0'])

<b> Output: </b>

![alt text]()

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

    # Freedom to Make Life Choices by Regions

    freedom_to_make_life_choices = data.groupby('regional_indicator')[['freedom_to_make_life_choices']].mean()
    freedom_to_make_life_choices

<b> Output: </b>

![alt text]()

    freedom_to_make_life_choices = freedom_to_make_life_choices.sort_values('freedom_to_make_life_choices',ascending = False)

    plt.rcParams['figure.figsize'] = (12,8)
    plt.title('Freedom to Make Life Choices in Various Regions')
    plt.xlabel('Regions',fontsize = 15)
    plt.ylabel('Freedom to Make Life Choices',fontsize = 15)
    plt.xticks(rotation = 30,ha = 'right')
    plt.bar(freedom_to_make_life_choices.index,freedom_to_make_life_choices.freedom_to_make_life_choices,color = ['C3','C6','C8','C4','C1','C2','C9','C5','C7','C0'])

<b> Output: </b>

![alt text]()

### STEP 7: Happiness Score and Perception of Corruption

    # Corruption and Happiness

    plt.rcParams['figure.figsize'] = (15,7)
    plt.title('Plot between Happiness and Corruption')
    sns.scatterplot(x = data.perceptions_of_corruption,y = data.happiness_score,hue = data.regional_indicator,s = 200)

    plt.legend(loc = 'upper left',fontsize = 14)
    plt.xlabel('Corruption Index')
    plt.ylabel('Happiness Score')
    
<b> Output: </b>

![alt text]()

    # Perception of Corruption by Regions

    perceptions_of_corruption = data.groupby('regional_indicator')[['perceptions_of_corruption']].mean()
    perceptions_of_corruption

<b> Output: </b>

![alt text]()

    perceptions_of_corruption = perceptions_of_corruption.sort_values('perceptions_of_corruption',ascending = False)

    plt.rcParams['figure.figsize'] = (12,8)
    plt.title('Perception of Corruption in Various Regions')
    plt.xlabel('Regions',fontsize = 15)
    plt.ylabel('Corruption Index',fontsize = 15)
    plt.xticks(rotation = 30,ha = 'right')
    plt.bar(perceptions_of_corruption.index,perceptions_of_corruption.perceptions_of_corruption,color = ['C3','C6','C8','C4','C1','C2','C9','C5','C7','C0'])

<b> Output: </b>

![alt text]()

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

    # Perception of Generosity by Regions

    generosity = data.groupby('regional_indicator')[['generosity']].mean()
    generosity

<b> Output: </b>

![alt text]()

    generosity = generosity.sort_values('generosity',ascending = False)

    plt.rcParams['figure.figsize'] = (12,8)
    plt.title('Generosity in Various Regions')
    plt.xlabel('Regions',fontsize = 15)
    plt.ylabel('Generosity',fontsize = 15)
    plt.xticks(rotation = 30,ha = 'right')
    plt.bar(generosity.index,generosity.generosity,color = ['C3','C6','C8','C4','C1','C2','C9','C5','C7','C0'])

<b> Output: </b>

![alt text]()

### STEP 9: Factors Contributing to Happiness - Regions

    # Factors Contributing to Happiness in Each Region

    regional_happiness = data.groupby('regional_indicator')[['happiness_score','dystopia','social_support','gdp_per_capita','healthy_life_expectancy','freedom_to_make_life_choices','perceptions_of_corruption','generosity']].mean()
    regional_happiness = regional_happiness.sort_values('happiness_score',ascending = False)

    plt.rcParams['figure.figsize'] = (12,8)
    plt.title('Factors Contributing to Happiness in Various Regions')
    plt.xlabel('Regions',fontsize = 15)
    plt.ylabel('Happiness Score',fontsize = 15)
    plt.xticks(rotation = 30,ha = 'right')

    plt.bar(regional_happiness.index,regional_happiness.dystopia,color = '#FF4040')
    plt.bar(regional_happiness.index,regional_happiness.gdp_per_capita,bottom = regional_happiness.dystopia,color = '#104E8B')
    plt.bar(regional_happiness.index,regional_happiness.social_support,bottom = regional_happiness.dystopia + regional_happiness.gdp_per_capita,color = '#00CDCD')
    plt.bar(regional_happiness.index,regional_happiness.healthy_life_expectancy,bottom = regional_happiness.dystopia + regional_happiness.gdp_per_capita + regional_happiness.social_support,color = '#A2CD5A')
    plt.bar(regional_happiness.index,regional_happiness.freedom_to_make_life_choices,bottom = regional_happiness.dystopia + regional_happiness.gdp_per_capita + regional_happiness.social_support + regional_happiness.healthy_life_expectancy,color = '#FFD700')
    plt.bar(regional_happiness.index,regional_happiness.perceptions_of_corruption,bottom = regional_happiness.dystopia + regional_happiness.gdp_per_capita + regional_happiness.social_support + regional_happiness.healthy_life_expectancy + regional_happiness.freedom_to_make_life_choices,color = '#B22222')
    plt.bar(regional_happiness.index,regional_happiness.generosity,bottom = regional_happiness.dystopia + regional_happiness.gdp_per_capita + regional_happiness.social_support + regional_happiness.healthy_life_expectancy + regional_happiness.freedom_to_make_life_choices + regional_happiness.perceptions_of_corruption,color = '#68228B')

<b> Output: </b>

![alt text]()

### STEP 10: Factors Contributing to Happiness - Top 10 Countries in the World

    # Top 10 Happiest Countries

    top_10 = data.head(10)

    plt.rcParams['figure.figsize'] = (12,8)
    plt.title('Factors Contributing to Happiness in Top 10 Happiest Countries')
    plt.xlabel('Countries',fontsize = 15)
    plt.ylabel('Happiness Score',fontsize = 15)
    plt.xticks(rotation = 30,ha = 'right')

    plt.bar(top_10.country,top_10.dystopia,color = '#FF4040')
    plt.bar(top_10.country,top_10.gdp_per_capita,bottom = top_10.dystopia,color = '#104E8B')
    plt.bar(top_10.country,top_10.social_support,bottom = top_10.dystopia + top_10.gdp_per_capita,color = '#00CDCD')
    plt.bar(top_10.country,top_10.healthy_life_expectancy,bottom = top_10.dystopia + top_10.gdp_per_capita + top_10.social_support,color = '#A2CD5A')
    plt.bar(top_10.country,top_10.freedom_to_make_life_choices,bottom = top_10.dystopia + top_10.gdp_per_capita + top_10.social_support + top_10.healthy_life_expectancy,color = '#FFD700')
    plt.bar(top_10.country,top_10.perceptions_of_corruption,bottom = top_10.dystopia + top_10.gdp_per_capita + top_10.social_support + top_10.healthy_life_expectancy + top_10.freedom_to_make_life_choices,color = '#B22222')
    plt.bar(top_10.country,top_10.generosity,bottom = top_10.dystopia + top_10.gdp_per_capita + top_10.social_support + top_10.healthy_life_expectancy + top_10.freedom_to_make_life_choices + top_10.perceptions_of_corruption,color = '#68228B')
    
<b> Output: </b>

![alt text]()

### STEP 11: Factors Contributing to Happiness - Bottom 10 Countries in the World

    # Bottom 10 Unhappiest Countries

    bottom_10 = data.tail(10)
    
    plt.rcParams['figure.figsize'] = (12,8)
    plt.title('Factors Contributing to Happiness in Top 10 Unhappiest Countries')
    plt.xlabel('Countries',fontsize = 15)
    plt.ylabel('Happiness Score',fontsize = 15)
    plt.xticks(rotation = 30,ha = 'right')

    plt.bar(bottom_10.country,bottom_10.dystopia,color = '#FF4040')
    plt.bar(bottom_10.country,bottom_10.gdp_per_capita,bottom = bottom_10.dystopia,color = '#104E8B')
    plt.bar(bottom_10.country,bottom_10.social_support,bottom = bottom_10.dystopia + bottom_10.gdp_per_capita,color = '#00CDCD')
    plt.bar(bottom_10.country,bottom_10.healthy_life_expectancy,bottom = bottom_10.dystopia + bottom_10.gdp_per_capita + bottom_10.social_support,color = '#A2CD5A')
    plt.bar(bottom_10.country,bottom_10.freedom_to_make_life_choices,bottom = bottom_10.dystopia + bottom_10.gdp_per_capita + bottom_10.social_support + bottom_10.healthy_life_expectancy,color = '#FFD700')
    plt.bar(bottom_10.country,bottom_10.perceptions_of_corruption,bottom = bottom_10.dystopia + bottom_10.gdp_per_capita + bottom_10.social_support + bottom_10.healthy_life_expectancy + bottom_10.freedom_to_make_life_choices,color = '#B22222')
    plt.bar(bottom_10.country,bottom_10.generosity,bottom = bottom_10.dystopia + bottom_10.gdp_per_capita + bottom_10.social_support + bottom_10.healthy_life_expectancy + bottom_10.freedom_to_make_life_choices + bottom_10.perceptions_of_corruption,color = '#68228B')

<b> Output: </b>

![alt text]()

## Insights:

1. 

## Conclusions:

1. 
