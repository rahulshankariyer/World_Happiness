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

A. North America and ANZ
B. Latin America and Carribean
C. Western Europe
D. Central and Eastern Europe
E. Commonwealth of Independent States
F. Middle East and North Africa
G. Sub-Saharan Africa
H. South Asia
I. Southeast Asia
J. East Asia

Total number of countries in each region:

    # Total Countries by Region

    total_country = data.groupby('regional_indicator')[['country']].count()
    total_country

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Total%20Countries%20by%20Region.png)

### STEP 2: Top 10 and Bottom 10 Countries on the Happiness Chart

    # Top 10 Happiest Countries

    top_10 = data.head(10)
    top_10
    
<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Top%2010%20Happiest%20Countries.png)

    # Bottom 10 Unhappiest Countries

    bottom_10 = data.tail(10)
    bottom_10

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Top%2010%20Unhappiest%20Countries.png)

### STEP 3: Correlation Map between Happiness and Six Different Influencers of Happiness

    # Correlation Map

    cor = data.corr(method = 'pearson')
    f, ax = plt.subplots(figsize = (10,5))
    sns.heatmap(cor, mask = np.zeros_like(cor,dtype = np.bool),cmap = 'Reds',square = True,ax = ax)
    
<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Correlation%20Map.png)

### STEP 4: Relationship between the Happiness Score and GDP Per Capita of the Countries, by Region

    # Plot between Happiness and GDP Per Capita

    plt.rcParams['figure.figsize'] = (15,7)
    plt.title('Plot between Happiness and GDP Per Capita')
    sns.scatterplot(x = data.happiness_score,y = data.gdp_per_capita,hue = data.regional_indicator,s = 200)

    plt.legend(loc = 'lower right',fontsize = '10')
    plt.xlabel('Happiness Score')
    plt.ylabel('GDP Per Capita')

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Plot%20Between%20Happiness%20and%20GDP%20Per%20Capita.png)

### STEP 5: Top 10 and Bottom 10 Countries by Happiness in Terms of GDP per capita

    # Top 10 Countries in terms of GDP Per Capita

    country = data.sort_values(by = 'gdp_per_capita').tail(10)
    plt.rcParams['figure.figsize'] = (12,6)
    plt.title('Top 10 Countries in terms of GDP Per Capita')
    plt.xlabel('Country',fontsize = 13)
    plt.ylabel('GDP Per Capita',fontsize = 13)
    plt.xticks(rotation = 30,ha = 'right')
    plt.bar(country.country,country.gdp_per_capita,color = 'green')

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Top%2010%20Countries%20in%20terms%20of%20GDP%20Per%20Capita.png)

    # Bottom 10 Countries in terms of GDP Per Capita

    country = data.sort_values(by = 'gdp_per_capita').head(10)
    plt.rcParams['figure.figsize'] = (12,6)
    plt.title('Bottom 10 Countries in terms of GDP Per Capita')
    plt.xlabel('Country',fontsize = 13)
    plt.ylabel('GDP Per Capita',fontsize = 13)
    plt.xticks(rotation = 30,ha = 'right')
    plt.bar(country.country,country.gdp_per_capita,color = 'red')

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Bottom%2010%20Countries%20in%20terms%20of%20GDP%20Per%20Capita.png)

### STEP 6: Relationship between the Happiness Score and Corruption Index of the Countries, by Region

    # Corruption and Happiness

    plt.rcParams['figure.figsize'] = (15,7)
    plt.title('Plot between Happiness and Corruption')
    sns.scatterplot(x = data.happiness_score,y = data.perceptions_of_corruption,hue = data.regional_indicator,s = 200)

    plt.legend(loc = 'upper left',fontsize = 14)
    plt.xlabel('Happiness Score')
    plt.ylabel('Corruption Index')
    
<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Plot%20Between%20Happiness%20and%20Corruption.png)

### STEP 7: Top 10 & Bottom 10 Countries in Terms of Perception of Corruption

    # Top 10 Countries in terms of Perception of Corruption

    country = data.sort_values(by = 'perceptions_of_corruption').tail(10)
    plt.rcParams['figure.figsize'] = (12,6)
    plt.title('Top 10 Countries in terms of Perception of Corruption')
    plt.xlabel('Country',fontsize = 13)
    plt.ylabel('Corruption Index',fontsize = 13)
    plt.xticks(rotation = 30,ha = 'right')
    plt.bar(country.country,country.perceptions_of_corruption,color = 'green')

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Top%2010%20Countries%20in%20terms%20of%20Perception%20of%20Corruption.png)

    # Bottom 10 Countries in terms of Perception of Corruption

    country = data.sort_values(by = 'perceptions_of_corruption').head(10)
    plt.rcParams['figure.figsize'] = (12,6)
    plt.title('Bottom 10 Countries in terms of Perception of Corruption')
    plt.xlabel('Country',fontsize = 13)
    plt.ylabel('Corruption Index',fontsize = 13)
    plt.xticks(rotation = 30,ha = 'right')
    plt.bar(country.country,country.perceptions_of_corruption,color = 'red')

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Bottom%2010%20Countries%20in%20terms%20of%20Perception%20of%20Corruption.png)

### STEP 8: Relationship between the Happiness Score and Healthy Life Expectancy of the Countries, by Region

    # Healthy Life Expectancy and Happiness

    plt.rcParams['figure.figsize'] = (15,7)
    plt.title('Plot between Happiness and Healthy Life Expectancy')
    sns.scatterplot(x = data.happiness_score,y = data.healthy_life_expectancy,hue = data.regional_indicator,s = 200)

    plt.legend(loc = 'upper left',fontsize = 14)
    plt.xlabel('Happiness Score')
    plt.ylabel('Healthy Life Expectancy')

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Plot%20between%20Happiness%20and%20Healthy%20Life%20Expectancy.png)

### STEP 9: Top 10 & Bottom 10 Countries in Terms of Healthy Life Expectancy

    # Top 10 Countries in terms of Healthy Life Expectancy

    country = data.sort_values(by = 'healthy_life_expectancy').tail(10)
    plt.rcParams['figure.figsize'] = (12,6)
    plt.title('10 Happiest Countries as per Healthy Life Expectancy')
    plt.xlabel('Country',fontsize = 13)
    plt.ylabel('Healthy Life Expectancy',fontsize = 13)
    plt.xticks(rotation = 30,ha = 'right')
    plt.bar(country.country,country.healthy_life_expectancy,color = 'green')

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Top%2010%20Countries%20in%20terms%20of%20Healthy%20Life%20Expectancy.png)

    # Bottom 10 Countries in terms of Healthy Life Expectancy

    country = data.sort_values(by = 'healthy_life_expectancy').head(10)
    plt.rcParams['figure.figsize'] = (12,6)
    plt.title('10 Happiest Countries as per Healthy Life Expectancy')
    plt.xlabel('Country',fontsize = 13)
    plt.ylabel('Healthy Life Expectancy',fontsize = 13)
    plt.xticks(rotation = 30,ha = 'right')
    plt.bar(country.country,country.healthy_life_expectancy,color = 'red')

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Bottom%2010%20Countries%20in%20terms%20of%20Healthy%20Life%20Expectancy.png)

### STEP 10: Happiness Score and Freedom to Make Life Choices of Countries, by Region

    # Freedom to Make Life Choices and Happiness

    plt.rcParams['figure.figsize'] = (15,7)
    plt.title('Plot between Happiness and Freedom to Make Life Choices')
    sns.scatterplot(x = data.freedom_to_make_life_choices,y = data.happiness_score,hue = data.regional_indicator,s = 200)

    plt.legend(loc = 'upper left',fontsize = '12')
    plt.xlabel('Freedom to Make Life Choices')
    plt.ylabel('Happiness Score')

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Plot%20between%20Happiness%20and%20Freedom%20to%20Make%20Life%20Choices.png)

### STEP 11: Top 10 & Bottom 10 Countries in Terms of Freedom to Make Life Choices

    # Top 10 Countries in terms of Freedom to Make Life Choices

    country = data.sort_values(by = 'freedom_to_make_life_choices').tail(10)
    plt.rcParams['figure.figsize'] = (12,6)
    plt.title('Top 10 Countries in terms of Freedom to Make Life Choices')
    plt.xlabel('Country',fontsize = 13)
    plt.ylabel('Freedom to Make Life Choices',fontsize = 13)
    plt.xticks(rotation = 30,ha = 'right')
    plt.bar(country.country,country.freedom_to_make_life_choices,color = 'green')

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Top%2010%20Countries%20in%20terms%20of%20Freedom%20to%20Make%20Life%20Choices.png)

    # Bottom 10 Countries in terms of Freedom to Make Life Choices

    country = data.sort_values(by = 'freedom_to_make_life_choices').head(10)
    plt.rcParams['figure.figsize'] = (12,6)
    plt.title('Bottom 10 Countries in terms of Freedom to Make Life Choices')
    plt.xlabel('Country',fontsize = 13)
    plt.ylabel('Freedom to Make Life Choices',fontsize = 13)
    plt.xticks(rotation = 30,ha = 'right')
    plt.bar(country.country,country.freedom_to_make_life_choices,color = 'red')

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Bottom%2010%20Countries%20in%20terms%20of%20Freedom%20to%20Make%20Life%20Choices.png)

### STEP 12: Happiness Score and Social Support of Countries, by Region

    # Social Support and Happiness

    plt.rcParams['figure.figsize'] = (15,7)
    plt.title('Plot between Happiness and Social Support')
    sns.scatterplot(x = data.social_support,y = data.happiness_score,hue = data.regional_indicator,s = 200)

    plt.legend(loc = 'upper left',fontsize = '12')
    plt.xlabel('Social Support')
    plt.ylabel('Happiness Score')

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Plot%20between%20Happiness%20and%20Social%20Support.png)

### STEP 13: Top 10 & Bottom 10 Countries in Terms of Social Support

    # Top 10 Countries in terms of Social Support

    country = data.sort_values(by = 'social_support').tail(10)
    plt.rcParams['figure.figsize'] = (12,6)
    plt.title('Top 10 Countries in terms of Social Support')
    plt.xlabel('Country',fontsize = 13)
    plt.ylabel('Social Support',fontsize = 13)
    plt.xticks(rotation = 30,ha = 'right')
    plt.bar(country.country,country.social_support,color = 'green')

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Top%2010%20Countries%20in%20terms%20of%20Social%20Support.png)

    # Bottom 10 Countries in terms of Social Support

    country = data.sort_values(by = 'social_support').head(10)
    plt.rcParams['figure.figsize'] = (12,6)
    plt.title('Bottom 10 Countries in terms of Social Support')
    plt.xlabel('Country',fontsize = 13)
    plt.ylabel('Social Support',fontsize = 13)
    plt.xticks(rotation = 30,ha = 'right')
    plt.bar(country.country,country.social_support,color = 'red')

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Bottom%2010%20Countries%20in%20terms%20of%20Social%20Support.png)

### STEP 14: Happiness Score and Generosity of Countries, by Region

    # Generosity and Happiness

    plt.rcParams['figure.figsize'] = (15,7)
    plt.title('Plot between Happiness and Generosity')
    sns.scatterplot(x = data.generosity,y = data.happiness_score,hue = data.regional_indicator,s = 200)

    plt.legend(loc = 'lower right',fontsize = '12')
    plt.xlabel('Generosity')
    plt.ylabel('Happiness Score')

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Plot%20between%20Happiness%20and%20Generosity.png)

### STEP 15: Top 10 & Bottom 10 Countries in Terms of Generosity

    # Top 10 Countries in terms of Generosity

    country = data.sort_values(by = 'generosity').tail(10)
    plt.rcParams['figure.figsize'] = (12,6)
    plt.title('Top 10 Countries in terms of Generosity')
    plt.xlabel('Country',fontsize = 13)
    plt.ylabel('Generosity',fontsize = 13)
    plt.xticks(rotation = 30,ha = 'right')
    plt.bar(country.country,country.generosity,color = 'green')

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Top%2010%20Countries%20in%20terms%20of%20Generosity.png)

    # Bottom 10 Countries in terms of Generosity

    country = data.sort_values(by = 'generosity').head(10)
    plt.rcParams['figure.figsize'] = (12,6)
    plt.title('Bottom 10 Countries in terms of Generosity')
    plt.xlabel('Country',fontsize = 13)
    plt.ylabel('Generosity',fontsize = 13)
    plt.xticks(rotation = 30,ha = 'right')
    plt.bar(country.country,country.generosity,color = 'red')

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Bottom%2010%20Countries%20in%20terms%20of%20Generosity.png)

## Insights:

1. 8 out of the Top 10 nations in terms of Happiness score are in Western Europe while 8 out of the Bottom 10 nations in terms of Happiness score are in Sub-Saharan Africa
2. A majority of the countries with the highest GDP are Western European countries. A closer look also shows us that the 4 countries from the North America and ANZ region - USA, Canada, Australia and NZ - are also near the top right, which indicates that they too have high GDP Per capita as well as Happiness Score
3. A majority of the countries with low GDP Per Capita as well as low Happiness score are from Africa and the Middle East
4. The Western European countries as well as North America and ANZ both had the best perception among its citizens when it comes to corruption while the Asian and African countries were among the worst in this regard
5. 7 out of the Top 10 countries in terms of Happiness score were also among the Top 10 best in terms of Perception of Corruption while Afghanistan and Lesotho were the only two nations in the Bottom 10 in both Happiness and Perception of Corruption.
6. 5 out of the 10 Unhappiest countries figured in the Bottom 10 in terms of Healthy Life Expectancy as well while Israel and Switzerland were in the only nations in the Top 10 in both Happiness and Healthy Life Expectancy
7. Western Europe, North America and ANZ gave the best Freedom to make life choices while Africa and the Middle East lacked in this regard
8. GDP Per Capita, Social Support and Healthy Life Expectancy had the biggest influence on the Happiness score of countries

## Conclusion:

1. Having a high GDP Per Capita, less Corruption and high degree of Freedom to make life choices definitely boosts the Happiness score
2. Having a low Healthy Life Expectancy definitely drags down the Happiness Score
3. Social support also had a major influence on the Happiness score while Generosity was the only indicator with little influence on the overall Happiness score
4. The Western countries as well as Australia and New Zealand are happier places to live in than Asia and Africa while South America and Eastern Europe are somewhere in the middle
