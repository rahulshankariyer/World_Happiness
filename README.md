# World Happiness

## Project Objective

To determine the impact of the Covid-19 pandemic on the Happiness of people across the world, compare the Happiness of different countries and also look specifically at how USA fared.

## Data Used

For the purpose of this analysis, we used the <a href = "https://www.kaggle.com/datasets/mathurinache/world-happiness-report?select=2022.csv"> World Happiness Data from 2015 to 2022 </a>. From this, only the data from 2020 to 2022 was taken, ie, the peak pandemic years.

## Tools Used

1. Excel
2. Python
3. Microsoft SQL Server Management Studio

## Data Extraction

Using Python, the required libraries were first imported.

    # Import the libraries

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
With the help of the pandas library that was imported, the CSV files containing the data were then extracted

    # Loading the datasets

    report_2020 = pd.read_csv('D:\\RAHUL JOBS\\Data Analytics Portfolio\\World Happiness\\2015-2022\\2020.csv')
    report_2021 = pd.read_csv('D:\\RAHUL JOBS\\Data Analytics Portfolio\\World Happiness\\2015-2022\\2021.csv')
    report_2022 = pd.read_csv('D:\\RAHUL JOBS\\Data Analytics Portfolio\\World Happiness\\2015-2022\\2022.csv')
    
## Data Overview

Before going into analysis, overview of the 3 dataframes that was created in the previous steps done

    # Overview of the data

    report_2020.head()
    report_2021.head()
    report_2022.head()
    
Here's a look at the columns present in the table

    # Investigating the columns

    print(report_2020.columns)
    print('nn')
    print(report_2021.columns)
    print('nn')
    print(report_2022.columns)

Output:

Index(['Country name', 'Regional indicator', 'Ladder score',
       'Standard error of ladder score', 'upperwhisker', 'lowerwhisker',
       'Logged GDP per capita', 'Social support', 'Healthy life expectancy',
       'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption', 'Ladder score in Dystopia',
       'Explained by: Log GDP per capita', 'Explained by: Social support',
       'Explained by: Healthy life expectancy',
       'Explained by: Freedom to make life choices',
       'Explained by: Generosity', 'Explained by: Perceptions of corruption',
       'Dystopia + residual'],
      dtype='object')
      
nn

Index(['Country name', 'Regional indicator', 'Ladder score',
       'Standard error of ladder score', 'upperwhisker', 'lowerwhisker',
       'Logged GDP per capita', 'Social support', 'Healthy life expectancy',
       'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption', 'Ladder score in Dystopia',
       'Explained by: Log GDP per capita', 'Explained by: Social support',
       'Explained by: Healthy life expectancy',
       'Explained by: Freedom to make life choices',
       'Explained by: Generosity', 'Explained by: Perceptions of corruption',
       'Dystopia + residual'],
      dtype='object')
      
nn

Index(['RANK', 'Country', 'Happiness score', 'Whisker-high', 'Whisker-low',
       'Dystopia (1#83) + residual', 'Explained by: GDP per capita',
       'Explained by: Social support', 'Explained by: Healthy life expectancy',
       'Explained by: Freedom to make life choices',
       'Explained by: Generosity', 'Explained by: Perceptions of corruption'],
      dtype='object')
      
## Data Cleaning

As part of the data cleaning process, the first step was to check for null values

    # Number of missing null values in each report

    print('2020 report:',report_2020.isnull().sum().sum())
    print('2021 report:',report_2021.isnull().sum().sum())
    print('2022 report:',report_2022.isnull().sum().sum())
    
Output:

2020 report: 0

2021 report: 0

2022 report: 10

As we can see from the above output, the 2022 data has 10 null values. So whichever rows containing null values were discarded.

    # Removing missing values from the 2022 report

    report_2022 = report_2022[report_2022.notnull().all(1)]
    report_2022
    
Now, let's quickly check the number of countries data used for each of these 3 years

    # Number of countries ranked in each report

    print('2020 report:',len(report_2020))
    print('2021 report:',len(report_2021))
    print('2022 report:',len(report_2022))
    
Output:

2020 report: 153

2021 report: 149

2022 report: 146

## Data Analysis

Let's have a look at the Top 5 countries in the world each year in terms of Happiness Score.

    # Top 5 Happiest Countries in each report

    plt.figure(figsize=(24,8))
    plt.subplot(1,3,1)
    sns.barplot(report_2020['Country name'][0:5],report_2020['Ladder score'][0:5])
    plt.subplot(1,3,2)
    sns.barplot(report_2021['Country name'][0:5],report_2021['Ladder score'][0:5])
    plt.subplot(1,3,3)
    sns.barplot(report_2022['Country'][0:5],report_2022['Happiness score'][0:5])
    
![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Top%205%20Happiest%20Countries.png)

With Finland topping all 3 reports, let's have a look at its Average Happiness Score.

    # Average score of Finland

    print('Average Happiness Score of Finland:',(report_2020['Ladder score'].iat[0] + report_2021['Ladder score'].iat[0] + report_2022['Happiness score'].iat[0]) / 3)
    
Output:

Average Happiness Score of Finland: 7.823900028333334

Now, let's move on to the Bottom 5 countries in the world each year in terms of Happiness Score.

    # Top 5 unhappiest countries in each report

    plt.figure(figsize=(24,8))
    plt.subplot(1,3,1)
    sns.barplot(report_2020['Country name'][148:153],report_2020['Ladder score'][148:153])
    plt.subplot(1,3,2)
    sns.barplot(report_2021['Country name'][144:149],report_2021['Ladder score'][144:149])
    plt.subplot(1,3,3)
    sns.barplot(report_2022['Country'][141:146],report_2022['Happiness score'][141:146])
    
![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Bottom%205%20Happiest%20Countries.png)

With Afghanistan at the bottom of all 3 reports, let's have a look at its Average Happiness Score.

    # Average score of Afghanistan

    print('Average Happiness Score of Afghanistan:',(report_2020['Ladder score'].iat[-1] + report_2021['Ladder score'].iat[-1] + report_2022['Happiness score'].iat[-1]) / 3)
    
Output:

Average Happiness Score of Afghanistan: 2.4979666716666666

Let's have a look at how USA has done during these 3 years and where they stand when it comes to overall Happiness Rankings. First, we will check out the rankings in terms of the overall Happiness Score.

    # Happiness Rankings of Countries in 2022

    import plotly.graph_objs as go
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 
    init_notebook_mode(connected=True)
    data = dict(type = 'choropleth', 
               locations = report_2022['Country'],
               locationmode = 'country names',
               z = report_2022['RANK'], 
               text = report_2022['Country'],
               colorbar = {'title':'Happiness'})
    layout = dict(title = 'Global Happiness 2022', 
                 geo = dict(showframe = False))
    choromap3 = go.Figure(data = [data], layout=layout)
    iplot(choromap3)
    
![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Happiness%20Rankings%202022.png)

For a more detailed analysis, we will now look at the rankings in each of the 6 indicators that contribute to the Happiness score

    # Countries Rankings based on GDP Per Capita in 2022

    import plotly.graph_objs as go
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 
    init_notebook_mode(connected=True)
    data = dict(type = 'choropleth', 
               locations = report_2022['Country'],
               locationmode = 'country names',
               z = report_2022['Explained by: GDP per capita'], 
               text = report_2022['Country'],
               colorbar = {'title':'GDP per capita'})
    layout = dict(title = 'GDP per capita in 2022', 
                 geo = dict(showframe = False))
    choromap3 = go.Figure(data = [data], layout=layout)
    iplot(choromap3)
    
![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/GDP%20Per%20Capita%20Rankings%202022.png)

    # Countries Rankings based on Healthy Life Expectancy in 2022

    import plotly.graph_objs as go
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 
    init_notebook_mode(connected=True)

    data = dict(type = 'choropleth', 
               locations = report_2022['Country'],
               locationmode = 'country names',
               z = report_2022['Explained by: Healthy life expectancy'], 
               text = report_2022['Country'],
               colorbar = {'title':'Healthy life expectancy'})
    layout = dict(title = 'Healthy life expectancy in 2022', 
                 geo = dict(showframe = False))
    choromap3 = go.Figure(data = [data], layout=layout)
    iplot(choromap3)

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Healthy%20Life%20Expectancy%20Rankings%202022.png)

    # Countries Rankings based on Social Support in 2022

    import plotly.graph_objs as go
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 
    init_notebook_mode(connected=True)

    data = dict(type = 'choropleth', 
               locations = report_2022['Country'],
               locationmode = 'country names',
               z = report_2022['Explained by: Social support'], 
               text = report_2022['Country'],
               colorbar = {'title':'Social support'})
    layout = dict(title = 'Social support in 2022', 
                 geo = dict(showframe = False))
    choromap3 = go.Figure(data = [data], layout=layout)
    iplot(choromap3)
    
![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Social%20Support%20Rankings%202022.png)

    # Countries Rankings based on Freedom to Make Life Choices in 2022

    import plotly.graph_objs as go
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 
    init_notebook_mode(connected=True)
    data = dict(type = 'choropleth', 
               locations = report_2022['Country'],
               locationmode = 'country names',
               z = report_2022['Explained by: Freedom to make life choices'], 
               text = report_2022['Country'],
               colorbar = {'title':'Freedom to make life choices in 2022'})
    layout = dict(title = 'Freedom to make life choices in 2022', 
                 geo = dict(showframe = False))
    choromap3 = go.Figure(data = [data], layout=layout)
    iplot(choromap3)
    
![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Freedom%20to%20Make%20Life%20Choices%20Rankings%202022.png)

    # Countries Rankings based on Generosity in 2022

    import plotly.graph_objs as go
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 
    init_notebook_mode(connected=True)
    data = dict(type = 'choropleth', 
               locations = report_2022['Country'],
               locationmode = 'country names',
               z = report_2022['Explained by: Generosity'], 
               text = report_2022['Country'],
               colorbar = {'title':'Generosity'})
    layout = dict(title = 'Generosity in 2022', 
                 geo = dict(showframe = False))
    choromap3 = go.Figure(data = [data], layout=layout)
    iplot(choromap3)
    
![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Generosity%20Rankings%202022.png)

    # Countries Rankings based on Perceptions of Corruption in 2022

    import plotly.graph_objs as go
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 
    init_notebook_mode(connected=True)
    data = dict(type = 'choropleth', 
               locations = report_2022['Country'],
               locationmode = 'country names',
               z = report_2022['Explained by: Perceptions of corruption'], 
               text = report_2022['Country'],
               colorbar = {'title':'Perceptions of corruption'})
    layout = dict(title = 'Perceptions of corruption in 2022', 
                 geo = dict(showframe = False))
    choromap3 = go.Figure(data = [data], layout=layout)
    iplot(choromap3)
    
![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Perceptions%20of%20Corruption%20Rankings%202022.png)

Let's see the Relationship of the Happiness Score with each of the 6 indicators in 2020 & 2021.

    # Relationships between Happiness Score and other data in all reports

    plt.figure(figsize=(24,8))

    plt.subplot(1,3,1)
    plt.scatter( report_2020['Ladder score'], report_2020['Logged GDP per capita'])
    plt.scatter( report_2021['Ladder score'], report_2021['Logged GDP per capita'])
    plt.xlabel('Logged GDP per capita in 2020 and 2021')
    plt.ylabel('Happiness score')
    plt.legend(['2020','2021'])

    plt.subplot(1,3,2)
    plt.scatter( report_2020['Ladder score'], report_2020['Social support'])
    plt.scatter( report_2021['Ladder score'], report_2021['Social support'])
    plt.xlabel('Social support in 2020 and 2021')
    plt.ylabel('Happiness score')
    plt.legend(['2020','2021'])

    plt.subplot(1,3,3)
    plt.scatter( report_2020['Ladder score'], report_2020['Healthy life expectancy'])
    plt.scatter( report_2021['Ladder score'], report_2021['Healthy life expectancy'])
    plt.xlabel('Healthy life expectancy in 2020 and 2021')
    plt.ylabel('Happiness score')
    plt.legend(['2020','2021'])

    plt.figure(figsize=(24,8))

    plt.subplot(1,3,1)
    plt.scatter( report_2020['Ladder score'], report_2020['Freedom to make life choices'])
    plt.scatter( report_2021['Ladder score'], report_2021['Freedom to make life choices'])
    plt.xlabel('Freedom to make life choices in 2020 and 2021')
    plt.ylabel('Happiness score')
    plt.legend(['2020','2021'])

    plt.subplot(1,3,2)
    plt.scatter( report_2020['Ladder score'], report_2020['Generosity'])
    plt.scatter( report_2021['Ladder score'], report_2021['Generosity'])
    plt.xlabel('Generosity in 2020 and 2021')
    plt.ylabel('Happiness score')
    plt.legend(['2020','2021'])

    plt.subplot(1,3,3)
    plt.scatter( report_2020['Ladder score'], report_2020['Perceptions of corruption'])
    plt.scatter( report_2021['Ladder score'], report_2021['Perceptions of corruption'])
    plt.xlabel('Perceptions of corruption in 2020 and 2021')
    plt.ylabel('Happiness score')
    plt.legend(['2020','2021'])
    
![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Happiness%20vs%20Each%20Indicator.png)

![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Happiness%20vs%20Each%20Indicator%202.png)

Let's now take a look at the correlation of Happiness Score and all the 6 indicators with each other in 2020 and 2021.

    # Correlations - 2020

    correlations_2020 = report_2020[['Logged GDP per capita', 'Social support', 'Healthy life expectancy',
                                     'Freedom to make life choices', 'Generosity','Perceptions of corruption',
                                     'Dystopia + residual']]
    sns.pairplot(correlations_2020, kind='reg')
    
![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Correlations%20-%202020.png)

    # Correlations - 2021

    correlations_2021 = report_2021[['Logged GDP per capita', 'Social support', 'Healthy life expectancy',
    'Freedom to make life choices', 'Generosity','Perceptions of corruption',
    'Dystopia + residual']]
    sns.pairplot(correlations_2021, kind='reg')
    
![alt text](https://raw.githubusercontent.com/rahulshankariyer/World_Happiness/main/Correlations%20-%202021.png)

## Insights

1. Finland, Denmark, Switzerland, Iceland, Netherlands and Norway were the happiest countries in the world overall from 2020 to 2022, with Finland at the top with 7.823900028333334
2. Afghanistan, Zimbabwe, Rwanda and Botswana were the unahppiest nations in the world overall from 2020 to 2022, with Afghanistan at the bottom with 2.4979666716666666
3. 
