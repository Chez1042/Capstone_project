# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load datasets
order_details = pd.read_csv('./Order Details.csv')
list_of_orders = pd.read_csv('./List of Orders.csv')

# Merge datasets
merged_data = pd.merge(order_details, list_of_orders, on='Order ID', how='inner')

# Data cleaning
merged_data['Order Date'] = pd.to_datetime(merged_data['Order Date'], errors='coerce')
merged_data.dropna(subset=['Order Date'], inplace=True)

# Add a month column for trend analysis
merged_data['Month'] = merged_data['Order Date'].dt.to_period('M')

# ---- 1. Analyze Sales Data ---- #
# Sales and profit trends over time
monthly_sales = merged_data.groupby('Month')[['Amount', 'Profit']].sum()

# Top categories and sub-categories
category_sales = merged_data.groupby('Category')['Amount'].sum().sort_values(ascending=False)
subcategory_sales = merged_data.groupby('Sub-Category')['Amount'].sum().sort_values(ascending=False)

# Regional sales distribution
regional_sales = merged_data.groupby('State')['Amount'].sum().sort_values(ascending=False)

# ---- 2. Perform Customer Segmentation ---- #
# Aggregate data by customer
customer_data = merged_data.groupby('CustomerName').agg(
    total_spent=('Amount', 'sum'),
    num_orders=('Order ID', 'count'),
    avg_order_value=('Amount', 'mean')
).reset_index()

# Prepare data for clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data[['total_spent', 'num_orders', 'avg_order_value']])

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
customer_data['Segment'] = kmeans.fit_predict(scaled_data)

# ---- 3. Develop Visualizations ---- #
sns.set_theme(style='whitegrid')

# 1. Sales trends over time
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales.index.astype(str), monthly_sales['Amount'], marker='o', label='Sales')
plt.plot(monthly_sales.index.astype(str), monthly_sales['Profit'], marker='o', label='Profit')
plt.title('Sales and Profit Trends Over Time')
plt.xlabel('Month')
plt.ylabel('Amount')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# 2. Top categories by sales
plt.figure(figsize=(8, 5))
sns.barplot(x=category_sales.values, y=category_sales.index, palette='viridis')
plt.title('Top Categories by Sales')
plt.xlabel('Total Sales')
plt.ylabel('Category')
plt.show()

# 3. Regional sales distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=regional_sales.values[:10], y=regional_sales.index[:10], palette='mako')
plt.title('Top 10 States by Sales')
plt.xlabel('Total Sales')
plt.ylabel('State')
plt.show()

# 4. Customer segmentation
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='total_spent', y='avg_order_value', hue='Segment', size='num_orders', data=customer_data,
    palette='cool', sizes=(20, 200)
)
plt.title('Customer Segmentation')
plt.xlabel('Total Spent')
plt.ylabel('Average Order Value')
plt.legend(title='Segment')
plt.show()

# 5. Profit distribution by category
plt.figure(figsize=(10, 6))
sns.boxplot(x='Category', y='Profit', data=merged_data, palette='pastel')
plt.title('Profit Distribution by Category')
plt.xlabel('Category')
plt.ylabel('Profit')
plt.show()

# Save the cleaned and analyzed data
merged_data.to_csv('./merged_data_analysis.csv', index=False)
customer_data.to_csv('./customer_segmentation.csv', index=False)