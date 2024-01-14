# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 19:06:38 2024

@author: TAMILSELVAN
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import seaborn as sns

# Function to read and filter data


def read_and_filter_data(file_path, indicator1, indicator2, year):
    """
    Read and filter data from a CSV file based on indicators and a specified
    year.

    Parameters:
    - file_path (str): Path to the CSV file.
    - indicator1 (str): First indicator name.
    - indicator2 (str): Second indicator name.
    - year (str): Year to filter data.

    Returns:
    - filtered_data (pd.DataFrame): Filtered and merged data.
    """
    df = pd.read_csv(file_path, skiprows=3)

    # Extract and rename columns for specified indicators and year
    data1 = df[df['Indicator Name'] == indicator1][[
        'Country Name', year]].rename(columns={year: indicator1})
    data2 = df[df['Indicator Name'] == indicator2][[
        'Country Name', year]].rename(columns={year: indicator2})

    # Merge dataframes on 'Country Name' and drop rows with any NaN values
    merged_data = pd.merge(data1, data2, on='Country Name',
                           how='outer').reset_index(drop=True)
    filtered_data = merged_data.dropna(how='any').reset_index(drop=True)
    return filtered_data

# Function to calculate inertia for KMeans clustering


def calculate_inertia(data, max_clusters=10):
    """
    Calculate inertia values for KMeans clustering with varying cluster numbers.

    Parameters:
    - data (pd.DataFrame): Input data for clustering.
    - max_clusters (int): Maximum number of clusters to consider.

    Returns:
    - inertias (list): List of inertia values for each cluster number.
    """
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    return inertias

# Function to cluster and plot subplots


def cluster_and_plot_subplot(
        df1, df2, cluster_columns, num_clusters=4, title1='', title2=''):
    """
    Apply KMeans clustering to two DataFrames and plot subplots for each with
    cluster information.

    Parameters:
    - df1 (pd.DataFrame): First DataFrame for clustering and plotting.
    - df2 (pd.DataFrame): Second DataFrame for clustering and plotting.
    - cluster_columns (list): Columns used for clustering.
    - num_clusters (int): Number of clusters to form.
    - title1 (str): Title for the first subplot.
    - title2 (str): Title for the second subplot.

    Returns:
    - None
    """
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Loop over dataframes, titles, and axes
    for df, title, ax in zip([df1, df2], [title1, title2], axes):
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(df[cluster_columns])
        df['cluster'] = df['cluster'].astype('category')

        # Get cluster centers
        cluster_centers = kmeans.cluster_centers_

        # Plot clusters
        scatter_plot = sns.scatterplot(
            x=cluster_columns[0], y=cluster_columns[1], hue='cluster', data=df,
            ax=ax)
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
                   marker='+', s=50, c='black', label='Cluster Centers')
        ax.set_title(title)

        # Create legend
        legend_handles = [Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=f'C{i}', markersize=8) 
                          for i in range(num_clusters)]
        legend_handles.append(
            Line2D([0], [0], marker='+', color='black', markersize=8))
        legend_labels = [
            f'Cluster {i + 1}' 
            for i in range(num_clusters)] + ['Cluster Centers']

        ax.legend(handles=legend_handles, labels=legend_labels,
                  loc='upper left', fontsize='small', handlelength=0.5,
                  handletextpad=0.5)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig('Forest_area_vs_CO2_emission_clusters.png')

# Function to fit polynomial regression model and plot with error range


def plot_with_error_range(X, y, degree, ax, title, color, actual_data_color):
    """
    Fit a polynomial regression model, plot the curve with an error range, and
    display actual data.

    Parameters:
    - X (pd.DataFrame): Input features for regression.
    - y (pd.Series): Target variable for regression.
    - degree (int): Degree of the polynomial regression.
    - ax (matplotlib.axes._subplots.AxesSubplot): Axes for plotting.
    - title (str): Title for the plot.
    - color (str): Color for the fitted curve.
    - actual_data_color (str): Color for the actual data points.

    Returns:
    - None
    """
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)

    # Initialize linear regression model
    model = LinearRegression()
    model.fit(X_poly, y)

    # Predict values for all years (1990 to 2025)
    X_pred = poly_features.transform(
        pd.DataFrame(all_years_extended, columns=['Year']))
    forecast_values = model.predict(X_pred)

    # Compute error range using bootstrapping
    n_bootstraps = 1000
    bootstrapped_predictions = np.zeros((n_bootstraps, len(X_pred)))

    # Iterate over bootstraps
    for i in range(n_bootstraps):
        indices = np.random.choice(len(X), len(X))
        X_bootstrapped = X.iloc[indices]
        y_bootstrapped = y.iloc[indices]

        X_poly_bootstrapped = poly_features.transform(X_bootstrapped)
        model.fit(X_poly_bootstrapped, y_bootstrapped)
        bootstrapped_predictions[i, :] = model.predict(X_pred)

    # Compute lower and upper bounds for error range
    lower_bound = np.percentile(bootstrapped_predictions, 2.5, axis=0)
    upper_bound = np.percentile(bootstrapped_predictions, 97.5, axis=0)

    # Plot actual data with a different color
    ax.plot(X, y, marker='.', linestyle='-',
            label='Actual Data', color=actual_data_color)

    # Plot fitted curve
    ax.plot(all_years_extended, forecast_values,
            label='Fitted Curve', linestyle='-', color=color)

    # Plot forecast for 2025
    prediction_2025 = forecast_values[-1]
    ax.plot(2025, prediction_2025, marker='o', markersize=8,
            label=f'Prediction for 2025: {prediction_2025:.2f}', color='black')

    # Plot error range
    ax.fill_between(all_years_extended, lower_bound, upper_bound,
                    color=color, alpha=0.3, label='95% Confidence Interval')

    # Set plot properties
    ax.set_title(title)
    ax.set_xlabel('Year')
    ax.set_ylabel('Kilotonns')
    ax.set_xlim(1990, 2030)
    ax.set_xticks(range(1990, 2031, 5))
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=7)


# Example usage
file_path = 'API_19_DS2_en_csv_v2_6300757.csv'
indicator1 = 'Forest area (sq. km)'
indicator2 = 'CO2 emissions (kt)'
year_2000 = '2000'
year_2020 = '2020'

# Read and filter data for 2000 and 2020
data_2000 = read_and_filter_data(file_path, indicator1, indicator2, year_2000)
data_2020 = read_and_filter_data(file_path, indicator1, indicator2, year_2020)

# Select relevant columns for clustering
X = data_2000[['Forest area (sq. km)', 'CO2 emissions (kt)']]
Y = data_2020[['Forest area (sq. km)', 'CO2 emissions (kt)']]

# Calculate inertia for the X DataFrame
inertias_X = calculate_inertia(X)

# Calculate inertia for the Y DataFrame
inertias_Y = calculate_inertia(Y)

# Create an elbow plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertias_X, marker='o', label=year_2000)
plt.plot(range(1, 11), inertias_Y, marker='o', label=year_2020)
plt.title('Elbow Plot for KMeans Clustering')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.legend()
plt.show()
# Define cluster_columns
cluster_columns = ['Forest area (sq. km)', 'CO2 emissions (kt)']

# Example usage of clustering and subplot plotting
cluster_and_plot_subplot(data_2000, data_2020, cluster_columns,
                         title1='Forest area and CO2 emission in 2000',
                         title2='Forest area and CO2 emission in 2020')

# Set a custom color palette for the plot
sns.set_palette("husl")

# Read data and filter for selected countries and indicator
df = pd.read_csv(file_path, skiprows=3)
selected_countries = ['China', 'India', 'United States']
indicator_name = 'CO2 emissions (kt)'
data_selected = df[(df['Country Name'].isin(selected_countries)) & (
    df['Indicator Name'] == indicator_name)].reset_index(drop=True)

# Melt the DataFrame
data_forecast = data_selected.melt(
    id_vars=['Country Name', 'Indicator Name'], var_name='Year',
    value_name='Value')

# Filter out non-numeric values in the 'Year' column
data_forecast = data_forecast[data_forecast['Year'].str.isnumeric()]

# Convert 'Year' to integers
data_forecast['Year'] = data_forecast['Year'].astype(int)

# Handle NaN values by filling with the mean value
data_forecast['Value'].fillna(data_forecast['Value'].mean(), inplace=True)

# Filter data for the years between 1990 and 2020
data_forecast = data_forecast[(data_forecast['Year'] >= 1990) & (
    data_forecast['Year'] <= 2020)]

# Create a dictionary to store predictions for each country
predictions = {}

# Extend the range of years to include 2025
all_years_extended = list(range(1990, 2026))

# Example usage of plotting with error range for selected countries
actual_data_colors = ['red', 'green', 'blue']
colors = ['mediumspringgreen', 'salmon', 'gold']

for country, color, actual_data_color, colors in zip(
        selected_countries, sns.color_palette("bright",
                                              len(selected_countries)),
        actual_data_colors, colors):
    fig, ax = plt.subplots(figsize=(6, 4))

    # Prepare data for the current country
    country_data = data_forecast[data_forecast['Country Name'] == country]
    X_country = country_data[['Year']]
    y_country = country_data['Value']

    # Plot with error range and different colors for actual data
    plot_with_error_range(X_country, y_country, degree=3, ax=ax,
                          title=f'{indicator_name} Forecast for {country}',
                          color=colors, actual_data_color=actual_data_color)

    # Save the figure with the title
    filename = f"{indicator_name}_Forecast_{country.replace(' ', '_')}.png"
    plt.savefig(filename, bbox_inches='tight')

    # Show the plot
    plt.show()
