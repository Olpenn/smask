import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('training_data_vt2025.csv')

df.iloc[:,15]=df.iloc[:,15].replace('low_bike_demand',False)
df.iloc[:,15]=df.iloc[:,15].replace('high_bike_demand',True)

import seaborn as sns

# MÅNADSBEROENDE



# Assuming your dataset has 'month' (1-12) and 'high_bike_demand' (True/False)
# Convert to numeric (if needed)
df['high_bike_demand'] = df['increase_stock'].astype(int)

# Calculate the percentage of high demand per month
monthly_demand = df.groupby('month')['increase_stock'].mean() * 100

# Plot
plt.figure(figsize=(10, 10))
sns.barplot(x=monthly_demand.index, y=monthly_demand.values, palette="Blues")

# Labels and Title
plt.xlabel("Month", fontsize=18)
plt.ylabel("Percentage of High Demand (%)", fontsize=18)
plt.title("Percentage of High Bike Demand per Month",fontsize=24)
plt.xticks(range(0, 12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=18)
#plt.gca().set_aspect('equal')
plt.savefig('demand_month.pdf',bbox_inches='tight')
plt.show()


# VECKODAGSBEROENDE


# Calculate the percentage of increase_stock per day of the week
weekly_stock_increase = df.groupby('day_of_week')['increase_stock'].mean() * 100

# Define labels for days of the week (assuming 0 = Monday, 6 = Sunday)
day_labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Plot using Seaborn
plt.figure(figsize=(10,10))
sns.barplot(x=weekly_stock_increase.index, y=weekly_stock_increase.values, palette="Blues")

# Labels and Title
plt.xlabel("Day of the Week", fontsize=18)
plt.ylabel("Percentage of Stock Increase (%)", fontsize=18)
plt.title("Percentage of Stock Increase per Day of the Week", fontsize=24)
plt.xticks(ticks=range(7), labels=day_labels)

# Show the plot
plt.savefig('demand_day.pdf',bbox_inches='tight')
plt.show()

# TIMME PÅ DYGNET

hourly_stock_increase = df.groupby('hour_of_day')['increase_stock'].mean() * 100

# Plot using Seaborn
plt.figure(figsize=(10, 3))
sns.barplot(x=hourly_stock_increase.index, y=hourly_stock_increase.values, palette="Blues")

# Labels and Title
plt.xlabel("Hour of the Day", fontsize=12)
plt.ylabel("Percentage of Stock Increase (%)", fontsize=12)
plt.title("Percentage of Stock Increase per Hour of the Day", fontsize=14)
plt.xticks(ticks=range(24))  # Ensure all hours are labeled

# Show the plot
plt.savefig('demand_hour.pdf',bbox_inches='tight')
plt.show()
'''
# Äntligen fick jag bra kod från ai, funkar så mycket bättre än alla 
# andra typ. Byt category och bin_no för att få olika plottar.
categories = ['dew','humidity','cloudcover','snowdepth','temp','visibility','precip']
for category in categories:
    # Define the number of bins and calculate bin edges
    bin_no = 10
    #category = categories[i]
    max_val = max(df[category])
    bin_edges = np.linspace(0, max_val, bin_no+1)
    right_bin_edges = bin_edges+0.01
    # Create IntervalIndex for categories
    categories = pd.IntervalIndex.from_arrays(bin_edges,
                                            right_bin_edges,
                                            closed='left')

    # Assign each value in {category} to a category using pd.cut
    df['Category'] = pd.cut(df[category],
                            bins=bin_edges,
                            include_lowest=True).rename('Category')

    # Group by the created category and compute the mean of 'increase_stock'
    grouped_data = df.groupby('Category')['increase_stock'].mean()

    # Create a bar plot with specified style
    plt.figure(figsize=(10, 10))
    grouped_data.plot(kind='bar', 
                    title=f'Mean increase_stock by {category}',
                    #rot_xticks=45, # Rotate x-tick labels for better readability
                    fontsize=18)

    # Add labels and title
    plt.xlabel(category)
    plt.ylabel(f'Mean increase_stock')
    plt.title(f'Mean increase_stock by {category}',fontsize=24)

    # Display the plot
    plt.savefig(f'demand_{category}.pdf',bbox_inches='tight')
    plt.show()
'''