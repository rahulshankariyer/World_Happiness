select * from ProjectPortfolio..[2022];

update ProjectPortfolio..[2022]
set [Happiness score] = [Happiness score] / 1000,
[Whisker-high] = [Whisker-high] / 1000,
[Whisker-low] = [Whisker-low] / 1000,
[Dystopia (1#83) + residual] = [Dystopia (1#83) + residual] / 1000,
[Explained by: GDP per capita] = [Explained by: GDP per capita] / 1000,
[Explained by: Social support] = [Explained by: Social support] / 1000,
[Explained by: Healthy life expectancy] = [Explained by: Healthy life expectancy] / 1000,
[Explained by: Freedom to make life choices] = [Explained by: Freedom to make life choices] / 1000,
[Explained by: Generosity] = [Explained by: Generosity] / 1000,
[Explained by: Perceptions of corruption] = [Explained by: Perceptions of corruption] / 1000;