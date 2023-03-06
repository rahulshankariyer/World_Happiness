# World Happiness

## Project Objective

To determine the impact of the Covid-19 pandemic on the Happiness of people across the world

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
