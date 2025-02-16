import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('training_data_vt2025.csv')

df.iloc[:,15]=df.iloc[:,15].replace('low_bike_demand',False)
df.iloc[:,15]=df.iloc[:,15].replace('high_bike_demand',True)

import seaborn as sns

# MÅNADSBEROENDE


'''
# Assuming your dataset has 'month' (1-12) and 'high_bike_demand' (True/False)
# Convert to numeric (if needed)
df['high_bike_demand'] = df['increase_stock'].astype(int)

# Calculate the percentage of high demand per month
monthly_demand = df.groupby('month')['increase_stock'].mean() * 100

# Plot
plt.figure(figsize=(10, 5))
sns.barplot(x=monthly_demand.index, y=monthly_demand.values, palette="Blues")

# Labels and Title
plt.xlabel("Month")
plt.ylabel("Percentage of High Demand (%)")
plt.title("Percentage of High Bike Demand per Month")
plt.xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])

plt.show()


# VECKODAGSBEROENDE


# Calculate the percentage of increase_stock per day of the week
weekly_stock_increase = df.groupby('day_of_week')['increase_stock'].mean() * 100

# Define labels for days of the week (assuming 0 = Monday, 6 = Sunday)
day_labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Plot using Seaborn
plt.figure(figsize=(8, 5))
sns.barplot(x=weekly_stock_increase.index, y=weekly_stock_increase.values, palette="Blues")

# Labels and Title
plt.xlabel("Day of the Week", fontsize=12)
plt.ylabel("Percentage of Stock Increase (%)", fontsize=12)
plt.title("Percentage of Stock Increase per Day of the Week", fontsize=14)
plt.xticks(ticks=range(7), labels=day_labels)

# Show the plot
plt.show()

# TIMME PÅ DYGNET

hourly_stock_increase = df.groupby('hour_of_day')['increase_stock'].mean() * 100

# Plot using Seaborn
plt.figure(figsize=(10, 5))
sns.barplot(x=hourly_stock_increase.index, y=hourly_stock_increase.values, palette="Blues")

# Labels and Title
plt.xlabel("Hour of the Day", fontsize=12)
plt.ylabel("Percentage of Stock Increase (%)", fontsize=12)
plt.title("Percentage of Stock Increase per Hour of the Day", fontsize=14)
plt.xticks(ticks=range(24))  # Ensure all hours are labeled

# Show the plot
plt.show()

# Snödjup, borde kunna modifieras för typ alla andra också.

snowdepth=np.sort(np.array(df['snowdepth'].unique()))
stock_per_snow = df.groupby('snowdepth')['increase_stock'].mean() * 100

plt.figure()
plt.plot(snowdepth,stock_per_snow)

plt.xlabel("Snowdepth (mm)", fontsize=12)
plt.ylabel("Percentage of Stock Increase (%)", fontsize=12)
plt.title("stock increase vs snowdepth", fontsize=14)
plt.show()
'''

# Vindhastighet
'''
# Define windspeed bins (adjust the range and step size as needed)
bin_edges = np.arange(0, df['windspeed'].max() + 10, 10)  # Bins of size 10 km/h

# Categorize the windspeed into bins
df['windspeed_bin'] = pd.cut(df['windspeed'], bins=bin_edges, right=False)

# Calculate the mean stock increase for each windspeed bin
stock_per_wind_bin = df.groupby('windspeed_bin')['increase_stock'].mean() * 100
stock_per_wind = df.groupby('increase_stock')['windspeed']
# Plotting the results
plt.figure(figsize=(10, 6))
stock_per_wind_bin.plot(kind='bar', color='skyblue')

# Adding labels and title
plt.xlabel("Windspeed Range (km/h)", fontsize=12)
plt.ylabel("Percentage of Stock Increase (%)", fontsize=12)
plt.title("Stock Increase vs Windspeed Range", fontsize=14)
plt.xticks(rotation=45)  # Rotate x labels for better readability

# Show the plot
plt.show()

plt.figure(figsize=(10, 6))
stock_per_wind.plot(kind='hist', color='skyblue',bins=10)

# Adding labels and title
plt.xlabel("Windspeed Range (km/h)", fontsize=12)
plt.ylabel("Percentage of Stock Increase (%)", fontsize=12)
plt.title("Stock Increase vs Windspeed Range", fontsize=14)
plt.xticks(rotation=45)  # Rotate x labels for better readability

# Show the plot
plt.show()
'''
# Precipitation
'''
# Define windspeed bins (adjust the range and step size as needed)
bin_edges = np.arange(0, df['precip'].max(), 1)  # Bins of size 10 km/h

# Categorize the windspeed into bins
df['precip_bin'] = pd.cut(df['precip'], bins=bin_edges, right=False)

# Calculate the mean stock increase for each windspeed bin
stock_per_wind_bin = df.groupby('precip_bin')['increase_stock'].mean() * 100

# Plotting the results
plt.figure(figsize=(10, 6))
stock_per_wind_bin.plot(kind='bar', color='skyblue')

# Adding labels and title
plt.xlabel("precipitaton Range (mm)", fontsize=12)
plt.ylabel("Percentage of Stock Increase (%)", fontsize=12)
plt.title("Stock Increase vs precipitation", fontsize=14)
plt.xticks(rotation=45)  # Rotate x labels for better readability

# Show the plot
plt.show()
'''

# Cloudcover

'''# Define windspeed bins (adjust the range and step size as needed)
bin_edges = np.arange(0, df['cloudcover'].max()+10, 10)  # Bins of size 10 km/h

# Categorize the windspeed into bins
df['cc_bin'] = pd.cut(df['cloudcover'], bins=bin_edges, right=False)

# Calculate the mean stock increase for each windspeed bin
stock_per_cc_bin = df.groupby('cc_bin')['increase_stock'].mean() * 100

# Plotting the results
plt.figure(figsize=(10, 6))
#stock_per_cc_bin.plot(kind='bar', color='skyblue')
stock_per_cc = df.groupby('cloudcover')['increase_stock'].mean() * 100
stock_per_cc.plot(kind='hist', color='skyblue')

# Adding labels and title
plt.xlabel("Cloud cover Range (%)", fontsize=12)
plt.ylabel("Percentage of Stock Increase (%)", fontsize=12)
plt.title("Stock Increase vs cloud cover", fontsize=14)
plt.xticks(rotation=45)  # Rotate x labels for better readability

# Show the plot
plt.show()
'''
# Snow

#print('it apparently never snows, according to the training data.')

# Visibility

# Categorize the windspeed into bins

# Calculate the mean stock increase for each windspeed bin
stock_per_cc = df.groupby('increase_stock')['visibility']
print(stock_per_cc)
# Plotting the results
plt.figure(figsize=(10, 6))
stock_per_cc.plot(kind='hist',
                  color='skyblue', 
                  bins=30,
                  )

# Adding labels and title
plt.xlabel("Cloud cover Range (%)", fontsize=12)
plt.ylabel("Percentage of Stock Increase (%)", fontsize=12)
plt.title("Stock Increase vs cloud cover", fontsize=14)
plt.xticks(rotation=45)  # Rotate x labels for better readability

# Show the plot
plt.show()
