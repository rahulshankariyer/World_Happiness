# World Happiness Data 2022

## Project Objective

Exploratory Data Analysis on the World Happiness Data of 146 countries, to explore the correlation between Happiness and factors influencing happiness such as Per Capita GDP, Freedom to Make Life Choices, Perceptions of Corruption, Life Expectancy, Social Support, and Generosity.

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

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/After%20Removing%20Columns.png)

    # Renaming columns

    happy_df = {'Country':'country','Regional indicator':'regional_indicator','Happiness score':'happiness_score','Dystopia (1#83) + residual':'dystopia','Explained by: GDP per capita':'gdp_per_capita','Explained by: Social support':'social_support','Explained by: Healthy life expectancy':'healthy_life_expectancy','Explained by: Freedom to make life choices':'freedom_to_make_life_choices','Explained by: Generosity':'generosity','Explained by: Perceptions of corruption':'perceptions_of_corruption'}
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

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Null%20Values.png)

Remove null values

    # Removing null values

    data = data[data.notnull().all(1)]
    data
    
<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Removing%20Null%20Values.png)

## Data Analysis Process

### STEP 1: Total Number of Countries in each Region
 
For this analysis, the countries were grouped into 10 different regions:

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

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Correlation%20Map.png)

    # Top Important Factors Affecting Happiness in 2022

    happiness_factors = cor['happiness_score'][2:]
    happiness_factors = happiness_factors.sort_values()
    happiness_factors
    
    happiness_factors_colors = {'social_support':'#00CDCD','gdp_per_capita':'#104E8B','healthy_life_expectancy':'#A2CD5A','freedom_to_make_life_choices':'#FFD700','perceptions_of_corruption':'#B22222','generosity':'#68228B'}

    colors = []
    for factor in happiness_factors.index:
        colors.append(happiness_factors_colors[factor])
    
    plt.rcParams['figure.figsize'] = (12,8)
    plt.title('Top Important Factors Affecting Happiness in 2022')
    plt.xlabel('Weightage',fontsize = 15)
    plt.barh(happiness_factors.index,happiness_factors,color = colors)
    
![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Top%20Important%20Factors%20Affecting%20Happiness%20in%202022.png)

### STEP 3: Happiness Score and Social Support

    # Social Support and Happiness

    plt.rcParams['figure.figsize'] = (15,7)
    plt.title('Plot between Happiness and Social Support')
    sns.scatterplot(x = data.social_support,y = data.happiness_score,hue = data.regional_indicator,s = 200)

    plt.legend(loc = 'lower right',fontsize = '12')
    plt.xlabel('Social Support')
    plt.gca().set_xlim([0,1.5])
    plt.ylabel('Happiness Score')
    plt.gca().set_ylim([0,8])

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Plot%20between%20Happiness%20and%20Social%20Support.png)

    # Social Support by Regions

    social_support = data.groupby('regional_indicator')[['social_support']].mean()
    social_support
    
<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Social%20Support%20by%20Regions.png)

    region_colours = {'Western Europe':'C0','Middle East and North Africa':'C1','North America and ANZ':'C2','Central and Eastern Europe':'C3','Latin America and Caribbean':'C4','Southeast Asia':'C5','Commonwealth of Independent States':'C6','Sub-Saharan Africa':'C7','East Asia':'C8','South Asia':'C9'}
    
    social_support = social_support.sort_values('social_support',ascending = False)

    colors = []
    for region in social_support.index:
        colors.append(region_colours[region])
    
    plt.rcParams['figure.figsize'] = (12,8)
    plt.title('Social Support in Various Regions')
    plt.ylabel('Social Support',fontsize = 15)
    plt.xticks(rotation = 30,ha = 'right')
    plt.bar(social_support.index,social_support.social_support,color = colors)

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Social%20Support%20in%20Various%20Regions.png)

### STEP 4: Happiness Score and GDP Per Capita

    # Plot between Happiness and GDP Per Capita

    plt.rcParams['figure.figsize'] = (15,7)
    plt.title('Plot between Happiness and GDP Per Capita')
    sns.scatterplot(x = data.gdp_per_capita,y = data.happiness_score,hue = data.regional_indicator,s = 200)

    plt.legend(loc = 'lower right',fontsize = '10')
    plt.xlabel('GDP Per Capita')
    plt.gca().set_xlim([0,2.5])
    plt.ylabel('Happiness Score')
    plt.gca().set_ylim([0,8])

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Plot%20Between%20Happiness%20and%20GDP%20Per%20Capita.png)

    # GDP Per Capita by Regions

    gdp_per_capita = data.groupby('regional_indicator')[['gdp_per_capita']].mean()
    gdp_per_capita

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/GDP%20Per%20Capita%20by%20Regions.png)

    gdp_per_capita = gdp_per_capita.sort_values('gdp_per_capita',ascending = False)
    
    colors = []
    for region in gdp_per_capita.index:
        colors.append(region_colours[region])
    
    plt.rcParams['figure.figsize'] = (12,8)
    plt.title('GDP Per Capita in Various Regions')
    plt.ylabel('GDP Per Capita',fontsize = 15)
    plt.xticks(rotation = 30,ha = 'right')
    plt.bar(gdp_per_capita.index,gdp_per_capita.gdp_per_capita,color = colors)

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/GDP%20Per%20Capita%20in%20Various%20Regions.png)

### STEP 5: Happiness Score and Healthy Life Expectancy

    # Healthy Life Expectancy and Happiness

    plt.rcParams['figure.figsize'] = (15,7)
    plt.title('Plot between Happiness and Healthy Life Expectancy')
    sns.scatterplot(x = data.healthy_life_expectancy,y = data.happiness_score,hue = data.regional_indicator,s = 200)

    plt.legend(loc = 'lower right',fontsize = 14)
    plt.xlabel('Healthy Life Expectancy')
    plt.gca().set_xlim([0,1])
    plt.ylabel('Happiness Score')
    plt.gca().set_ylim([0,8])

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Plot%20between%20Happiness%20and%20Healthy%20Life%20Expectancy.png)

    # Healthy Life Expectancy by Regions

    healthy_life_expectancy = data.groupby('regional_indicator')[['healthy_life_expectancy']].mean()
    healthy_life_expectancy

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Healthy%20Life%20Expectancy%20by%20Regions.png)

    healthy_life_expectancy = healthy_life_expectancy.sort_values('healthy_life_expectancy',ascending = False)
    
    colors = []
    for region in healthy_life_expectancy.index:
        colors.append(region_colours[region])
    
    plt.rcParams['figure.figsize'] = (12,8)
    plt.title('Healthy Life Expectancy in Various Regions')
    plt.ylabel('Healthy Life Expectancy',fontsize = 15)
    plt.xticks(rotation = 30,ha = 'right')
    plt.bar(healthy_life_expectancy.index,healthy_life_expectancy.healthy_life_expectancy,color = colors)

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Healthy%20Life%20Expectancy%20in%20Various%20Regions.png)

### STEP 6: Happiness Score and Freedom to Make Life Choices

    # Freedom to Make Life Choices and Happiness

    plt.rcParams['figure.figsize'] = (15,7)
    plt.title('Plot between Happiness and Freedom to Make Life Choices')
    sns.scatterplot(x = data.freedom_to_make_life_choices,y = data.happiness_score,hue = data.regional_indicator,s = 200)

    plt.legend(loc = 'lower right',fontsize = '12')
    plt.xlabel('Freedom to Make Life Choices')
    plt.gca().set_xlim([0,0.8])
    plt.ylabel('Happiness Score')
    plt.gca().set_ylim([0,8])

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Plot%20between%20Happiness%20and%20Freedom%20to%20Make%20Life%20Choices.png)

    # Freedom to Make Life Choices by Regions

    freedom_to_make_life_choices = data.groupby('regional_indicator')[['freedom_to_make_life_choices']].mean()
    freedom_to_make_life_choices

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Freedom%20to%20Make%20Life%20Choices%20by%20Regions.png)

    freedom_to_make_life_choices = freedom_to_make_life_choices.sort_values('freedom_to_make_life_choices',ascending = False)
    
    colors = []
    for region in freedom_to_make_life_choices.index:
        colors.append(region_colours[region])
    
    plt.rcParams['figure.figsize'] = (12,8)
    plt.title('Freedom to Make Life Choices in Various Regions')
    plt.ylabel('Freedom to Make Life Choices',fontsize = 15)
    plt.xticks(rotation = 30,ha = 'right')
    plt.bar(freedom_to_make_life_choices.index,freedom_to_make_life_choices.freedom_to_make_life_choices,color = colors)

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Freedom%20to%20Make%20Life%20Choices%20in%20Various%20Regions.png)

### STEP 7: Happiness Score and Perception of Corruption

    # Corruption and Happiness

    plt.rcParams['figure.figsize'] = (15,7)
    plt.title('Plot between Happiness and Corruption')
    sns.scatterplot(x = data.perceptions_of_corruption,y = data.happiness_score,hue = data.regional_indicator,s = 200)

    plt.legend(loc = 'lower right',fontsize = 14)
    plt.xlabel('Absence of Corruption')
    plt.gca().set_xlim([0,0.6])
    plt.ylabel('Happiness Score')
    plt.gca().set_ylim([0,8])
    
<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Plot%20Between%20Happiness%20and%20Corruption.png)

    # Absence of Corruption by Regions

    perceptions_of_corruption = data.groupby('regional_indicator')[['perceptions_of_corruption']].mean()
    perceptions_of_corruption

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Perception%20of%20Corruption%20by%20Regions.png)

    perceptions_of_corruption = perceptions_of_corruption.sort_values('perceptions_of_corruption',ascending = False)
    
    colors = []
    for region in perceptions_of_corruption.index:
        colors.append(region_colours[region])
    
    plt.rcParams['figure.figsize'] = (12,8)
    plt.title('Absence of Corruption in Various Regions')
    plt.ylabel('Absence of Corruption',fontsize = 15)
    plt.xticks(rotation = 30,ha = 'right')
    plt.bar(perceptions_of_corruption.index,perceptions_of_corruption.perceptions_of_corruption,color = colors)

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Absence%20of%20Corruption%20in%20Various%20Regions.png)

### STEP 8: Happiness Score and Generosity

    # Generosity and Happiness

    plt.rcParams['figure.figsize'] = (15,7)
    plt.title('Plot between Happiness and Generosity')
    sns.scatterplot(x = data.generosity,y = data.happiness_score,hue = data.regional_indicator,s = 200)

    plt.legend(loc = 'lower right',fontsize = '12')
    plt.xlabel('Generosity')
    plt.gca().set_xlim([0,0.5])
    plt.ylabel('Happiness Score')
    plt.gca().set_ylim([0,8])

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Plot%20between%20Happiness%20and%20Generosity.png)

    # Generosity by Regions

    generosity = data.groupby('regional_indicator')[['generosity']].mean()
    generosity

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Generosity%20by%20Regions.png)

    generosity = generosity.sort_values('generosity',ascending = False)
    
    colors = []
    for region in generosity.index:
        colors.append(region_colours[region])
    
    plt.rcParams['figure.figsize'] = (12,8)
    plt.title('Generosity in Various Regions')
    plt.ylabel('Generosity',fontsize = 15)
    plt.xticks(rotation = 30,ha = 'right')
    plt.bar(generosity.index,generosity.generosity,color = colors)

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Generosity%20in%20Various%20Regions.png)

### STEP 9: Factors Contributing to Happiness - Regions

    # Factors Contributing to Happiness in Each Region

    regional_happiness = data.groupby('regional_indicator')[['happiness_score','dystopia','social_support','gdp_per_capita','healthy_life_expectancy','freedom_to_make_life_choices','perceptions_of_corruption','generosity']].mean()
    regional_happiness = regional_happiness.sort_values('happiness_score',ascending = False)

    plt.rcParams['figure.figsize'] = (12,8)
    plt.title('Factors Contributing to Happiness in Various Regions')
    plt.ylabel('Happiness Score',fontsize = 15)
    plt.xticks(rotation = 30,ha = 'right')

    plt.bar(regional_happiness.index,regional_happiness.dystopia,color = '#CD5B45')
    plt.bar(regional_happiness.index,regional_happiness.gdp_per_capita,bottom = regional_happiness.dystopia,color = '#104E8B')
    plt.bar(regional_happiness.index,regional_happiness.social_support,bottom = regional_happiness.dystopia + regional_happiness.gdp_per_capita,color = '#00CDCD')
    plt.bar(regional_happiness.index,regional_happiness.healthy_life_expectancy,bottom = regional_happiness.dystopia + regional_happiness.gdp_per_capita + regional_happiness.social_support,color = '#A2CD5A')
    plt.bar(regional_happiness.index,regional_happiness.freedom_to_make_life_choices,bottom = regional_happiness.dystopia + regional_happiness.gdp_per_capita + regional_happiness.social_support + regional_happiness.healthy_life_expectancy,color = '#FFD700')
    plt.bar(regional_happiness.index,regional_happiness.perceptions_of_corruption,bottom = regional_happiness.dystopia + regional_happiness.gdp_per_capita + regional_happiness.social_support + regional_happiness.healthy_life_expectancy + regional_happiness.freedom_to_make_life_choices,color = '#B22222')
    plt.bar(regional_happiness.index,regional_happiness.generosity,bottom = regional_happiness.dystopia + regional_happiness.gdp_per_capita + regional_happiness.social_support + regional_happiness.healthy_life_expectancy + regional_happiness.freedom_to_make_life_choices + regional_happiness.perceptions_of_corruption,color = '#68228B')
    
    plt.legend(['Dystopia','GDP Per Capita','Social Support','Healthy Life Expectancy','Freedom to Make Life Choices','Perceptions of Corruption','Generosity'],fontsize = 10)

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Factors%20Contributing%20to%20Happiness%20in%20Various%20Regions.png)

### STEP 10: Factors Contributing to Happiness - Top 10 Countries in the World

    # 10 Happiest Countries

    top_10 = data.head(10)

    plt.rcParams['figure.figsize'] = (12,8)
    plt.title('Factors Contributing to Happiness in 10 Happiest Countries')
    plt.ylabel('Happiness Score',fontsize = 15)
    plt.xticks(rotation = 30,ha = 'right')

    plt.bar(top_10.country,top_10.dystopia,color = '#CD5B45')
    plt.bar(top_10.country,top_10.gdp_per_capita,bottom = top_10.dystopia,color = '#104E8B')
    plt.bar(top_10.country,top_10.social_support,bottom = top_10.dystopia + top_10.gdp_per_capita,color = '#00CDCD')
    plt.bar(top_10.country,top_10.healthy_life_expectancy,bottom = top_10.dystopia + top_10.gdp_per_capita + top_10.social_support,color = '#A2CD5A')
    plt.bar(top_10.country,top_10.freedom_to_make_life_choices,bottom = top_10.dystopia + top_10.gdp_per_capita + top_10.social_support + top_10.healthy_life_expectancy,color = '#FFD700')
    plt.bar(top_10.country,top_10.perceptions_of_corruption,bottom = top_10.dystopia + top_10.gdp_per_capita + top_10.social_support + top_10.healthy_life_expectancy + top_10.freedom_to_make_life_choices,color = '#B22222')
    plt.bar(top_10.country,top_10.generosity,bottom = top_10.dystopia + top_10.gdp_per_capita + top_10.social_support + top_10.healthy_life_expectancy + top_10.freedom_to_make_life_choices + top_10.perceptions_of_corruption,color = '#68228B')
    
    plt.legend(['Dystopia','GDP Per Capita','Social Support','Healthy Life Expectancy','Freedom to Make Life Choices','Perceptions of Corruption','Generosity'],fontsize = 10,loc = 'upper right')
    
<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Factors%20Contributing%20to%20Happiness%20in%20Top%2010%20Happiest%20Countries.png)

### STEP 11: Factors Contributing to Happiness - Bottom 10 Countries in the World

    # 10 Unhappiest Countries

    bottom_10 = data.tail(10)
    
    plt.rcParams['figure.figsize'] = (12,8)
    plt.title('Factors Contributing to Happiness in 10 Unhappiest Countries')
    plt.ylabel('Happiness Score',fontsize = 15)
    plt.xticks(rotation = 30,ha = 'right')

    plt.bar(bottom_10.country,bottom_10.dystopia,color = '#CD5B45')
    plt.bar(bottom_10.country,bottom_10.gdp_per_capita,bottom = bottom_10.dystopia,color = '#104E8B')
    plt.bar(bottom_10.country,bottom_10.social_support,bottom = bottom_10.dystopia + bottom_10.gdp_per_capita,color = '#00CDCD')
    plt.bar(bottom_10.country,bottom_10.healthy_life_expectancy,bottom = bottom_10.dystopia + bottom_10.gdp_per_capita + bottom_10.social_support,color = '#A2CD5A')
    plt.bar(bottom_10.country,bottom_10.freedom_to_make_life_choices,bottom = bottom_10.dystopia + bottom_10.gdp_per_capita + bottom_10.social_support + bottom_10.healthy_life_expectancy,color = '#FFD700')
    plt.bar(bottom_10.country,bottom_10.perceptions_of_corruption,bottom = bottom_10.dystopia + bottom_10.gdp_per_capita + bottom_10.social_support + bottom_10.healthy_life_expectancy + bottom_10.freedom_to_make_life_choices,color = '#B22222')
    plt.bar(bottom_10.country,bottom_10.generosity,bottom = bottom_10.dystopia + bottom_10.gdp_per_capita + bottom_10.social_support + bottom_10.healthy_life_expectancy + bottom_10.freedom_to_make_life_choices + bottom_10.perceptions_of_corruption,color = '#68228B')
    
    plt.legend(['Dystopia','GDP Per Capita','Social Support','Healthy Life Expectancy','Freedom to Make Life Choices','Perceptions of Corruption','Generosity'],fontsize = 10,loc = 'upper right')

<b> Output: </b>

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Factors%20Contributing%20to%20Happiness%20in%20Top%2010%20Unhappiest%20Countries.png)

## Insights:

1. The top 3 factors that influence happiness are, in order, GDP Per Capita, Social Support and Healthy Life Expectancy.
2. Surprisingly, Freedom to Make Life Choices ranks 4th, and had little (no) correlation with any other factor.
3. Absence of Corruption & Generosity played a negligible role in influencing happiness.
4. The highest correlation was between GDP Per Capita and Healthy Life Expectancy.

## Notes:

1. The insights are for Happiness Worldwide.
2. Since the Original Data had already ranked the nations, no ranking, by country or region, was discussed as an insight.
3. The data on Generosity may be inaccurate as the question asked on it pertained to "an act of generosity or charity yesterday"

## Conclusions:

1. GDP Per Capita, Social Support, and Healthy Life Expectancy being basic needs, and highly correlated, the developed/affluent nations fared well on the Happiness Chart.
2. While Freedom to Make Life Choices was more important in countries that had a tradition of Liberal Democracy, countries that never had freedoms do not think of it as contributor to happiness.
3. Absence of Corruption scored low in most countries probably because of the following:

    a) People in developed countries have come to expect it as a routine.
    
    b) In many countries, people have accepted it, or are ignorant about it.
