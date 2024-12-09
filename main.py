""" Load dataset and importing libraries """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from scipy.stats.contingency import association

df = pd.read_csv('amz_uk_price_prediction_dataset.csv')


""" Part 1: Analyzing Best-Seller Trends Across Product Categories """

# 1. Crosstab Analysis:
#crosstab_result = pd.crosstab(df['category'], df['isBestSeller'])

# 2. Statistical Tests:
# Chi-square test for 'MSZoning' and 'SaleCondition'
#_, chi2_pvalue, _, _  = chi2_contingency(crosstab_result)
#print(float(chi2_pvalue) < 0.05)
# Computing the association between variables in 'crosstab_result' using the "cramer" method
#print(association(crosstab_result, method='cramer'))

# 3. Visualizations:
# Plotting a stacked bar chart for the 'crosstab_result' data
#crosstab_result.head().plot(kind='bar', stacked=True, figsize=(12,6))
#plt.show()

""" Part 2: Exploring Product Prices and Ratings Across Categories and Brands """
# 0. Preliminary Step: Remove outliers in product prices.
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for the outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_without_outliers = df[(df['price'] > lower_bound) | (df['price'] < upper_bound)]

top_cat = df_without_outliers[['price','category']].groupby('category').count().nlargest(5)

print(top_cat)

# 1. Violin Plots:
plt.figure(figsize=(12,8))
sns.violinplot(data=top_cat, x="category", y="price", palette="coolwarm")
plt.show()