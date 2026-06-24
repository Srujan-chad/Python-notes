DADV LAB Programs

1a. Develop a Python program to demonstrate the fundamental functionalities of NumPy, Pandas, and Matplotlib in a single workflow.
The program should:
•	NumPy:
1.	Create a 2D NumPy array.
2.	Display the array.
3.	Find its shape and data type.
4.	Calculate the mean of the array.
5.	Find the transpose of the array.
•  Pandas:
1.	Create a DataFrame containing Name, Age, City, and Department.
2.	Display the DataFrame.
3.	Show DataFrame information.
4.	Generate summary statistics of the dataset.
•  Matplotlib:
1.	Generate values from 1 to 5.
2.	Compute their squares.
3.	Plot a simple line graph with title, axis labels, markers, and grid.
4.	Plot a bar chart to represent age distribution.
# Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create a 2D NumPy Array
arr = np.array([[10, 20, 30],
                [40, 50, 60]])

# Display Array
print("NumPy Array:\n", arr)

# Array Properties and Operations
print("Shape of Array:", arr.shape)
print("Data Type of Array:", arr.dtype)
print("Mean of Array:", np.mean(arr))
print("Transpose of Array:\n", arr.T)

# Create a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['Bangalore', 'Chennai', 'Mumbai'],
    'Department': ['CSE', 'ECE', 'ISE']
}

df = pd.DataFrame(data)

# Display DataFrame
print("DataFrame:\n", df)

# Display DataFrame Information
print("\nDataFrame Info:")
df.info()

# Generate Summary Statistics
print("\nDescriptive Statistics:\n", df.describe(include='all'))


# Matplotlib Demonstration
# Generate Values from 1 to 5
x = np.arange(1, 6)

# Compute Squares
y = x ** 2

# Line Plot
plt.figure(figsize=(6, 4))
plt.plot(x, y, marker='o', linestyle='-', label='y = x²')
plt.title("Line Plot: y = x²")
plt.xlabel("X Values")
plt.ylabel("Squared Values")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Bar Chart 
names = ['Alice', 'Bob', 'Charlie']
ages = [24, 29, 35]

plt.figure(figsize=(6, 4))
plt.bar(names, ages)
plt.title("Bar Chart: Age Distribution")
plt.xlabel("Name")
plt.ylabel("Age")
plt.tight_layout()
plt.show()

2a. Write a Python program to demonstrate data cleaning and preprocessing techniques using the Titanic dataset. The program should perform the following operations:
1.	Load the dataset using Pandas.
2.	Identify and handle missing values in the dataset.
3.	Remove duplicate records if present.
4.	Encode categorical variables into numerical format.
5.	Apply feature scaling to numerical columns.
6.	Display the cleaned dataset and summary statistics.
# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 2: Load the dataset
df = pd.read_csv('titanic.csv')

print("Initial Data:")
print(df.head())

print("\nMissing Values in Dataset:")
print(df.isnull().sum())

# Step 3: Handle Missing Values
# Fill missing Age values with mean
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Drop rows where Embarked value is missing
df.dropna(subset=['Embarked'], inplace=True)

# Replace missing Cabin values with 'Unknown'
df['Cabin'].fillna('Unknown', inplace=True)

print("\nAfter Handling Missing Values:")
print(df.isnull().sum())

# Step 4: Remove Duplicate Records
initial_rows = df.shape[0]
df.drop_duplicates(inplace=True)
final_rows = df.shape[0]

print("\nNumber of duplicates removed:", initial_rows - final_rows)

# Step 5: Encode Categorical Variables

# Convert 'Sex' column into numeric values
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# One-Hot Encoding for 'Embarked'
df = pd.get_dummies(df, columns=['Embarked'])

# Drop unnecessary columns
df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Step 6: Feature Scaling
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Step 7: Display Final Dataset
print("\nCleaned Data Preview:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

2b. Write a Python program to explore and analyze a dataset using the Pandas and NumPy libraries. The program should perform the following tasks:
1.	Load the dataset from a CSV file.
2.	Display the first and last few records of the dataset.
3.	Show the structure and information of the dataset.
4.	Generate a statistical summary of numerical columns.
5.	Display the column names in the dataset.
6.	Check for missing (null) values in each column.
7.	Analyze categorical data such as unique products and city-wise order counts.
8.	Use NumPy functions to calculate statistical measures like mean, median, and standard deviation for a numerical column.
# Step 1: Import required libraries
import pandas as pd
import numpy as np

# Step 2: Load the dataset
df = pd.read_csv('sample_sales.csv')

# Step 3: Preview the dataset
print("----- First 5 Rows -----")
print(df.head())

print("\n----- Last 5 Rows -----")
print(df.tail())

# Step 4: Dataset structure
print("\n----- DataFrame Info -----")
print(df.info())

# Step 5: Statistical summary
print("\n----- Statistical Summary -----")
print(df.describe())

# Step 6: Column names
print("\n----- Column Names -----")
print(df.columns)

# Step 7: Null value check
print("\n----- Null Values in Each Column -----")
print(df.isnull().sum())

# Step 8: Categorical column insights
print("\n----- Unique Products -----")
print(df['Product'].unique())

print("\n----- City-wise Order Count -----")
print(df['City'].value_counts())

# Step 9: NumPy statistics on numerical column

# Clean and convert data
df['Quantity Ordered'] = pd.to_numeric(df['Quantity Ordered'], errors='coerce')
quantities = df['Quantity Ordered'].dropna()

print("\n----- Quantity Ordered Statistics -----")
print("Mean:", np.mean(quantities))
print("Median:", np.median(quantities))
print("Standard Deviation:", np.std(quantities))

# Optional: Save cleaned dataset
# df.to_csv('cleaned_sales_data.csv', index=False)

3.Write a Python program using Pandas to perform data aggregation and group operations on a sales dataset.
The program should group the data and calculate summary statistics such as:
1.	Total quantity ordered for each product.
2.	Average quantity ordered in each city.
3.	Total revenue generated in each city.
4.	Minimum, maximum, and mean quantity ordered for each product using multiple aggregation functions.
Use groupby(), agg(), and sorting functions to analyze the dataset.

# Step 1: Import required libraries
import pandas as pd
import numpy as np

# Step 2: Load the dataset
df = pd.read_csv('cleaned_sales_data.csv')

print("First 5 rows of dataset:")
print(df.head())

# Step 3: Convert columns to numeric
df['Quantity Ordered'] = pd.to_numeric(df['Quantity Ordered'], errors='coerce')
df['Price Each'] = pd.to_numeric(df['Price Each'], errors='coerce')

# Step 4: Create Revenue column
df['Revenue'] = df['Quantity Ordered'] * df['Price Each']

# Step 5: Total quantity ordered per product
product_qty = df.groupby('Product')['Quantity Ordered'].sum().reset_index()

print("\nTotal Quantity Ordered per Product:")
print(product_qty)

# Step 6: Average quantity ordered per city
city_avg_qty = df.groupby('City')['Quantity Ordered'].mean().reset_index()

print("\nAverage Quantity Ordered per City:")
print(city_avg_qty)

# Step 7: Total revenue per city
city_revenue = df.groupby('City')['Revenue'].sum().reset_index()

# Sort cities by revenue
city_revenue = city_revenue.sort_values(by='Revenue', ascending=False)

print("\nTotal Revenue per City:")
print(city_revenue)

# Step 8: Multiple aggregation on Quantity Ordered
product_stats = df.groupby('Product')['Quantity Ordered'].agg(['min', 'max', 'mean']).reset_index()

print("\nMinimum, Maximum and Mean Quantity per Product:")
print(product_stats)

4a. Write a Python program to analyze relationships between multiple variables in a dataset using correlation analysis and pairwise visualization techniques.
he program should:
1.	Load a dataset using Python.
2.	Compute the correlation matrix for numerical variables.
3.	Visualize correlations using a heatmap.
4.	Generate pairwise plots (pairplots) to observe relationships between variables.
5.	Interpret the correlation values to understand positive, negative, or no relationship between variables.
Use the Iris dataset and Titanic dataset available in the Seaborn library to perform the analysis.
# Experiment 4: Correlation Analysis and Pairwise Analysis of Multivariate Data

# Step 1: Import Required Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set visual style
sns.set(style="whitegrid")

# Step 2: Load the Dataset

# Iris Dataset
iris = sns.load_dataset('iris')
print("Iris Dataset Preview:")
print(iris.head())

# Titanic Dataset
titanic = sns.load_dataset('titanic')
print("\nTitanic Dataset Preview:")
print(titanic[['age', 'fare', 'pclass', 'survived']].head())

# Step 3: Compute Correlation Matrix for Iris Dataset
correlation_iris = iris.corr(numeric_only=True)

print("\nCorrelation Matrix - Iris Dataset:")
print(correlation_iris)

# Step 4: Visualize Correlation using Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(correlation_iris, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap - Iris Dataset")
plt.show()

# Step 5: Pairplot for Iris Dataset
sns.pairplot(iris, hue='species')
plt.suptitle("Pairplot - Iris Dataset", y=1.02)
plt.show()

# Step 6: Clean Titanic Dataset (select numerical columns)
titanic_clean = titanic[['age', 'fare', 'pclass', 'survived']].dropna()

# Step 7: Compute Correlation Matrix for Titanic Dataset
correlation_titanic = titanic_clean.corr()

print("\nCorrelation Matrix - Titanic Dataset:")
print(correlation_titanic)

# Step 8: Heatmap for Titanic Dataset
plt.figure(figsize=(8,6))
sns.heatmap(correlation_titanic, annot=True, cmap='YlGnBu', linewidths=0.5)
plt.title("Correlation Heatmap - Titanic Dataset")
plt.show()

# Step 9: Pairplot for Titanic Dataset
sns.pairplot(titanic_clean, hue='survived', palette="husl")
plt.suptitle("Pairplot - Titanic Dataset", y=1.02)
plt.show()



4b. Write a Python program to perform correlation analysis and pairwise visualization on the Wine Quality dataset to understand the relationships between different chemical properties of wine and the wine quality score.
The program should:
1.	Load the Wine Quality dataset from a CSV file.
2.	Display the first few rows of the dataset.
3.	Check for missing values in the dataset.
4.	Compute the correlation matrix for numerical attributes.
5.	Visualize the correlation using a heatmap.
6.	Generate pairwise plots (pairplots) for selected features to observe relationships between variables and wine quality.
# Correlation and Pairwise Analysis using Wine Quality Dataset

# Step 1: Import Required Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Set Seaborn Style
sns.set(style="whitegrid")

# Step 3: Load the Wine Quality Dataset
# The dataset should be downloaded and saved as 'winequality-red.csv'
# The delimiter used in this dataset is ';'

wine = pd.read_excel(“winequality_red.xlsx")

# Step 4: Display Dataset Preview
print("Wine Dataset Preview:")
print(wine.head())

# Step 5: Check for Missing Values
print("\nMissing Values:\n", wine.isnull().sum())

# Step 6: Generate Correlation Matrix
corr_matrix = wine.corr(numeric_only=True)
print("\nCorrelation Matrix - Wine Quality Dataset:")
print(corr_matrix)

# Step 7: Visualize Correlation Matrix using Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap - Wine Quality Dataset")
plt.tight_layout()
plt.show()

# Step 8: Generate Pairplot for Selected Features
selected_features = ['alcohol', 'sulphates', 'citric acid', 'residual sugar', 'quality']

sns.pairplot(wine[selected_features], hue='quality', palette='Set2', diag_kind='kde')
plt.suptitle("Pairwise Plot - Wine Quality Dataset", y=1.02)
plt.show()

5.  Using a given sales dataset (cleaned_sales_data.csv), perform data visualization to analyze patterns and insights by creating the following charts using Matplotlib:
1.	📊 Plot a Bar Chart to show total quantity ordered per product. 
2.	📈 Plot a Line Chart to display revenue generated across different cities. 
3.	📉 Plot a Histogram to visualize the distribution of quantity ordered. 
4.	🥧 Plot a Pie Chart to represent the proportion of total revenue contributed by each city.
#  Step 1: Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt

#  Step 2: Load Dataset
df = pd.read_csv("cleaned_sales_data.csv")
# Convert columns to numeric (handle errors safely)
df['Quantity Ordered'] = pd.to_numeric(df['Quantity Ordered'], errors='coerce')
df['Price Each'] = pd.to_numeric(df['Price Each'], errors='coerce')
# Create Revenue column
df['Revenue'] = df['Quantity Ordered'] * df['Price Each']

#  Step 3: Aggregated Data
# Total quantity per product
product_qty = df.groupby('Product')['Quantity Ordered'].sum().reset_index()

# Total revenue per city
city_revenue = df.groupby('City')['Revenue'].sum().reset_index()
# Quantity data for histogram
quantity_data = df['Quantity Ordered'].dropna()

# Step 4: Bar Chart – Quantity by Product
plt.figure(figsize=(10, 6))
plt.bar(product_qty['Product'], product_qty['Quantity Ordered'], color='skyblue')
plt.title('Total Quantity Ordered per Product')
plt.xlabel('Product')
plt.ylabel('Quantity Ordered')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 5: Line Chart – Revenue by City
plt.figure(figsize=(8, 5))
plt.plot(city_revenue['City'], city_revenue['Revenue'], 
         marker='o', linestyle='--', color='green')
plt.title('Revenue by City')
plt.xlabel('City')
plt.ylabel('Revenue')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 6: Histogram – Quantity Distribution
plt.figure(figsize=(8, 5))
plt.hist(quantity_data, bins=20, color='orange', edgecolor='black')
plt.title('Distribution of Quantity Ordered')
plt.xlabel('Quantity Ordered')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 7: Pie Chart – Revenue Proportion by City
plt.figure(figsize=(7, 7))
plt.pie(city_revenue['Revenue'], 
        labels=city_revenue['City'], 
        autopct='%1.1f%%', 
        startangle=140)
plt.title('Proportion of Revenue by City')
plt.tight_layout()
plt.show()


6a. Using a cleaned sales dataset (cleaned_sales_data.csv), perform advanced data visualization using Seaborn to analyze relationships and distributions:
1.	📊 Create a Scatter Plot for Quantity Ordered vs Revenue 
2.	📦 Create a Box Plot for Revenue by City 
3.	🔥 Create a Heatmap for correlation between numerical features 
4.	🔗 Create a Pair Plot for selected numerical variables
# Step 1: Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load dataset
df = pd.read_csv("cleaned_sales_data.csv")

# Convert to numeric
df['Quantity Ordered'] = pd.to_numeric(df['Quantity Ordered'], errors='coerce')
df['Price Each'] = pd.to_numeric(df['Price Each'], errors='coerce')

# Create Revenue column
df['Revenue'] = df['Quantity Ordered'] * df['Price Each']

# Step 3: Scatter Plot
plt.figure(figsize=(8,6))
sns.scatterplot(x='Quantity Ordered', y='Revenue', data=df)
plt.title('Quantity Ordered vs Revenue')
plt.xlabel('Quantity Ordered')
plt.ylabel('Revenue')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 4: Box Plot
plt.figure(figsize=(8,6))
sns.boxplot(x='City', y='Revenue', data=df)
plt.title('Revenue Distribution by City')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 5: Heatmap
corr = df[['Quantity Ordered', 'Price Each', 'Revenue']].corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Step 6: Pair Plot
sns.pairplot(df[['Quantity Ordered', 'Price Each', 'Revenue']])
plt.suptitle('Pair Plot of Features', y=1.02)
plt.tight_layout()
plt.show()


6b. Using the built-in Titanic dataset from Seaborn, perform advanced visualizations to understand survival patterns and feature relationships:
1.	 Create a Pair Plot for numerical variables with survival status 
2.	Create a Correlation Heatmap 
3.	Create a Violin Plot for age distribution by gender 
4.	Create a Swarm Plot for fare distribution by class 
5.	Create a Categorical Plot (CatPlot) for fare by gender across classes 
6.	Create a Facet Grid for age distribution by class and gender
# Step 1: Import required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load Titanic dataset
df = sns.load_dataset('titanic')

# Clean dataset (remove missing values)
df = df[['sex', 'age', 'fare', 'class', 'pclass', 'survived']].dropna()

# Step 3: Pair Plot
sns.pairplot(df[['age', 'fare', 'pclass', 'survived']], hue='survived')
plt.suptitle('Pair Plot of Titanic Dataset', y=1.02)
plt.show()

# Step 4: Heatmap
correlation_matrix = df[['age', 'fare', 'pclass', 'survived']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Step 5: Violin Plot
sns.violinplot(x='sex', y='age', data=df)
plt.title("Violin Plot of Age by Sex")
plt.show()

# Step 6: Swarm Plot
sns.swarmplot(x='class', y='fare', data=df)
plt.title("Swarm Plot of Fare by Class")
plt.show()

# Step 7: Cat Plot
sns.catplot(x='sex', y='fare', col='class', kind='box', data=df)
plt.suptitle("Cat Plot: Fare by Sex Across Classes", y=1.05)
plt.show()

# Step 8: Facet Grid
g = sns.FacetGrid(df, col="class", row="sex")
g.map_dataframe(sns.histplot, x="age", bins=20)
g.set_titles(col_template="{col_name} Class", row_template="{row_name}")
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Age Distribution by Class and Sex")
plt.show()


7.	The goal of this experiment is to design and develop interactive data visualizations using the Plotly library in Python.
The experiment focuses on:
•	Creating different types of charts such as bar charts, line charts, and scatter plots 
•	Enabling interactive features like zooming, panning, and hover tooltips 
•	Customizing visualizations for better readability and presentation 
•	Exporting visualizations as HTML files for sharing and embedding
# Import Libraries
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# step 1: Bar Chart (Population by Continent)

# Load dataset
df = px.data.gapminder().query("year == 2007")

# Create Bar Chart
fig_bar = px.bar(
    df,
    x='continent',
    y='pop',
    color='continent',
    title='Population by Continent'
)

# Display Bar Chart
fig_bar.show()


# step 2: Line Chart (India GDP Trend)

# Filter data for India
df_line = px.data.gapminder().query("country == 'India'")

# Create Line Chart
fig_line = px.line(
    df_line,
    x='year',
    y='gdpPercap',
    title='India GDP per Capita Over Years',
    markers=True
)

# Display Line Chart
fig_line.show()


# step 3: Scatter Plot (GDP vs Life Expectancy)

# Create Scatter Plot
fig_scatter = px.scatter(
    df,
    x="gdpPercap",
    y="lifeExp",
    size="pop",
    color="continent",
    hover_name="country",
    log_x=True,
    size_max=60,
    title="GDP vs Life Expectancy (2007)"
)

# Customize Layout
fig_scatter.update_layout(
    xaxis_title="GDP Per Capita (log scale)",
    yaxis_title="Life Expectancy",
    legend_title="Continent"
)

# Display Scatter Plot
fig_scatter.show()


# step 4: Export Visualization

# Export scatter plot to HTML file
fig_scatter.write_html("scatter_plot.html")


# step 5: Additional Example (COVID-19 Bar Chart)

# Sample COVID dataset
covid_data = {
    'Country': ['USA', 'India', 'Brazil', 'UK', 'Russia'],
    'Cases': [33000000, 31000000, 19000000, 6000000, 5000000],
    'Deaths': [600000, 410000, 530000, 130000, 120000]
}

# Convert to DataFrame
df_covid = pd.DataFrame(covid_data)

# Create Bar Chart
fig_covid = px.bar(
    df_covid,
    x='Country',
    y='Cases',
    color='Country',
    title='COVID-19 Total Cases by Country',
    hover_data=['Deaths'],
    text='Cases'
)

# Customize Layout
fig_covid.update_layout(
    xaxis_title='Country',
    yaxis_title='Total Cases',
    template='plotly_white'
)

# Display Chart
fig_covid.show()



8. An e-commerce company wants to analyze its historical transaction data to understand sales trends, customer behavior, and revenue distribution across different product categories and countries.
You are given a dataset (ecommerce_sales.csv) containing order details such as order date, product category, sales amount, and country.
Using this data:
•	Process and clean the dataset 
•	Perform time-based and categorical analysis 
•	Visualize trends using graphs 
•	Derive meaningful business insights from the data 

🗂️ Input Dataset Format
OrderID	OrderDate	CustomerID	ProductCategory	Amount	Country
1001	2022-01-03	C123	Electronics	24999	India
1002	2022-01-04	C124	Fashion	1999	USA

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Style settings
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Load the dataset
df = pd.read_csv('ecommerce_sales.csv')

# Display basic info
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# Check column names (important for debugging)
print("\nColumn Names:")
print(df.columns)

# Convert order_date to datetime (FIXED)
df['orderDate'] = pd.to_datetime(df['order_date'], dayfirst=True)

# Extract date parts
df['Year'] = df['orderDate'].dt.year
df['Month'] = df['orderDate'].dt.month
df['Month_Name'] = df['orderDate'].dt.strftime('%b')

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Handle missing values
df.dropna(inplace=True)

# Monthly Sales Aggregation
monthly_sales = df.groupby(['Year', 'Month_Name'])['Amount'].sum().reset_index()

# Convert Month_Name to Month number for sorting
monthly_sales['Month_Num'] = pd.to_datetime(monthly_sales['Month_Name'], format='%b').dt.month
monthly_sales = monthly_sales.sort_values(by=['Year', 'Month_Num'])

# Line Plot – Monthly Sales Trend
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_sales, x='Month_Name', y='Amount', hue='Year', marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales Amount')
plt.legend(title='Year')
plt.tight_layout()
plt.show()

# Sales by Product Category 
category_sales = df.groupby('product_category')['Amount'].sum().sort_values(ascending=False)

# Bar Plot – Sales by Category
plt.figure(figsize=(10, 5))
sns.barplot(x=category_sales.index, y=category_sales.values)
plt.title('Sales by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Top 5 Countries by Revenue
top_countries = df.groupby('Country')['Amount'].sum().sort_values(ascending=False).head(5)

# Pie Chart – Revenue Share by Country
plt.figure(figsize=(8, 6))
top_countries.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title('Top 5 Countries by Revenue')
plt.ylabel('')
plt.tight_layout()
plt.show()

# Heatmap – Product Category vs Month 
pivot = df.pivot_table(values='Amount',
                       index='product_category',
                       columns='Month_Name',
                       aggfunc='sum').fillna(0)

# Correct month order
ordered_months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
pivot = pivot.reindex(columns=ordered_months)

# Heatmap Plot
plt.figure(figsize=(12, 6))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap='YlGnBu')
plt.title('Sales Heatmap (Product Category vs Month)')
plt.xlabel('Month')
plt.ylabel('Product Category')
plt.tight_layout()
plt.show()



9 Apply Principal Component Analysis (PCA) on the Iris dataset to reduce the number of features from four to two principal components. Standardize the dataset before applying PCA, and visualize the transformed data to observe the distribution of different classes.

# Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Load the Dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map(dict(enumerate(iris.target_names)))

# Step 2: Separate features and target
X = df.iloc[:, :4]
y = df['species']

# Step 3: Standardize the Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 5: Create DataFrame with PCA results
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['species'] = y.values

# Step 6: Print Explained Variance Ratio
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Step 7: Visualize PCA Components
plt.figure(figsize=(10,6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='species', palette='Set1')
plt.title('PCA: First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

# Step 8: Scree Plot (Cumulative Explained Variance)
pca_full = PCA()
pca_full.fit(X_scaled)

plt.figure(figsize=(8,5))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o')
plt.title('Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()


10 Perform customer segmentation using the K-Means clustering algorithm on the Mall_Customers dataset. The goal is to group customers based on their purchasing behavior using features such as Annual Income and Spending Score.
Use the Elbow Method to determine the optimal number of clusters and visualize the final customer segments using appropriate plots.
# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('Mall_Customers.csv')

# Display first few rows
print(df.head())

# Select relevant features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Feature scaling (standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method to find optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Apply KMeans with optimal k (assume k = 5)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='Set2',
    data=df,
    s=100
)

plt.title('Customer Segmentation using K-Means')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Display dataset with cluster labels
print(df.head(10))
