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

Here's an overview of the 3 dataframes that we created in the previous step:

