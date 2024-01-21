#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 21:39:24 2024

@author: diya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.optimize as opt
from sklearn.cluster import KMeans

# Function to read and process data from an Excel file
def process(raw_data, skip):
    """
    Read and process data from an Excel file.

    Parameters:
    - raw_data (str): The filename of the Excel file.
    - skip (int): Number of rows to skip when reading the Excel file.

    Returns:
    - dt_data (DataFrame): The raw data as a pandas DataFrame.
    - dt_transpose (DataFrame): The transposed version of the raw data.
    """
    dt_data = pd.read_excel(raw_data, skiprows=skip)
    dt_transpose = dt_data.transpose()
    return dt_data, dt_transpose

# Function to process indicator data for a specific year
def process_indicator_data(dt_data, indicator_name, data_columns, drop_columns):
    """
    Process indicator data for a specific indicator name.

    Parameters:
    - dt_data (DataFrame): The raw data as a pandas DataFrame.
    - indicator_name (str): The name of the indicator to extract.
    - data_columns (list): List of columns to keep in the resulting DataFrame.
    - drop_columns (list): List of columns to drop from the resulting DataFrame.

    Returns:
    - indicator_data (DataFrame): Processed indicator data as a pandas DataFrame.
    """
    indicator_data = dt_data[dt_data['Indicator Name'] == indicator_name]
    indicator_data = (indicator_data.drop(drop_columns, axis=1)
                      .filter(data_columns)
                      .dropna()
                      .set_index('Country Name'))
    return indicator_data

# Function to normalize data
def normalize_data(data):
    """
    Normalize data between 0 and 1.

    Parameters:
    - data (DataFrame): Data to be normalized.

    Returns:
    - normalized_data (array): Normalized data.
    """
    values = data[['Total Green Gas emissions', 
                   'GDP per capita growth (annual %)']].values
    max_value = values.max(axis=0)  # Max value for each column
    min_value = values.min(axis=0)  # Min value for each column
    
    if np.all(max_value == min_value):
        print("Normalization Error: Max and min values are the same for a feature.")
        return None
    
    normalized_data = (values - min_value) / (max_value - min_value)
    return normalized_data

# Function to define a polynomial function
def polynomial(x, a, b, c):
    """
    Polynomial function used for curve fitting.

    Parameters:
    - x (array): Input values.
    - a, b, c (float): Polynomial coefficients.

    Returns:
    - y (array): Output values.
    """
    return a * x**2 + b * x + c

# Function to calculate error for curve fitting
def get_error(x, y, degree):
    """
    Calculate the standard deviation of residuals for curve fitting.

    Parameters:
    - x (array): Input values.
    - y (array): Actual values.
    - degree (int): Degree of the polynomial.

    Returns:
    - error (float): Standard deviation of residuals.
    """
    coefficients = np.polyfit(x, y, degree)
    y_estimate = np.polyval(coefficients, x)
    residuals = y - y_estimate
    return np.std(residuals)

# Function to plot data points
def plot_data(year, emissions, gdp, ax, color):
    """
    Plot data points for a specific year.

    Parameters:
    - year (str): The year for which the data is being plotted.
    - emissions (Series): Greenhouse gas emissions data.
    - gdp (Series): GDP per capita growth data.
    - ax (Axes): The matplotlib axes to plot on.
    - color (str): Color for the data points.
    """
    ax.scatter(emissions, gdp, c=color)
    ax.set_title(year, fontweight='bold', fontsize=12)
    ax.set_xlabel(
        'Total greenhouse gas emissions (kt of CO2 equivalent)',
        fontweight='bold')
    ax.set_ylabel('GDP per capita growth (annual %)', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)

# Function to perform K-Means clustering and plot clusters
def perform_kmeans_clustering(data, year, ax, colors=['red', 'green', 'blue']):
    """
    Perform K-Means clustering and plot clusters for a specific year.

    Parameters:
    - data (DataFrame): Data for clustering.
    - year (str): The year for which clustering is performed.
    - ax (Axes): The matplotlib axes to plot on.
    - colors (list): List of colors for cluster points.

    Returns:
    - values_norm (array): Normalized data for clustering.
    - labels (array): Cluster labels assigned to data points.
    """
    values = data[['Total Green Gas emissions',
                   'GDP per capita growth (annual %)']].values
    max_value = values.max(axis=0)
    min_value = values.min(axis=0)
    
    # Here we add a check to ensure that max_value is not equal to min_value for any feature
    if (max_value == min_value).any():
        print("Normalization Error: Max and min values are the same for at least one feature.")
        return None, None
    
    values_norm = (values - min_value) / (max_value - min_value)

    ncluster = 3
    kmeans = KMeans(n_clusters=ncluster, init='k-means++', max_iter=300,
                    n_init=10, random_state=0)
    kmeans.fit(values_norm)
    labels = kmeans.labels_

    # Map the cluster labels to colors
    color_map = [colors[label] for label in labels]
    ax.scatter(values_norm[:, 0], values_norm[:, 1], c=color_map)
    ax.set_title(f'K-Means Clustering {year}', fontweight='bold', fontsize=12)
    ax.set_xlabel('Total greenhouse gas emissions (kt of CO2 equivalent)', 
                  fontweight='bold')
    ax.set_ylabel('GDP per capita growth (annual %)', fontweight='bold')
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.grid(True, linestyle='--', alpha=0.7)
    return values_norm, labels

# Function to perform clustering for multiple years and plot clusters
def n_clustering_and_plotting(values_norm, year, axs):
    """
    Perform clustering for multiple years and plot clusters.

    Parameters:
    - values_norm (array): Normalized data for clustering.
    - year (str): The year for which clustering is performed.
    - axs (Axes): The matplotlib axes to plot on.

    Returns:
    - predictions (array): Cluster labels assigned to data points.
    """
    # Check if values_norm is not None
    if values_norm is None:
        print(f"Skipping plotting for year {year} due to normalization error.")
        return None

    # Check the range of data to make sure there is variance
    if np.all(values_norm[:, 0] == values_norm[0, 0]):
        print(f"All x-axis values are the same for year {year}: {values_norm[0, 0]}")
        return None

    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300,
                    n_init=10, random_state=0)
    kmeans.fit(values_norm)
    predictions = kmeans.predict(values_norm)
    cluster_centers = kmeans.cluster_centers_

    colors = ['darkred', 'lightcoral', 'orange', 'goldenrod']
    for cluster in range(n_clusters):
        axs.scatter(values_norm[predictions == cluster, 0],
                    values_norm[predictions == cluster, 1], s=50, 
                    c=colors[cluster], label=f'cluster {cluster}')
    axs.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=50,
                c='black', label='Centroids', marker='+')
    axs.set_title(f'Country clusters - Emissions/GDP : {year}', 
                  fontweight='bold', fontsize=12)
    axs.set_xlabel('Total greenhouse gas emissions (kt of CO2 equivalent)',
                   fontweight='bold')
    axs.set_ylabel('GDP per capita growth (annual %)', fontweight='bold')
    axs.legend()
    axs.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    axs.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'n_clusters_{year}.png')
    return predictions

# Function to plot the elbow method for K-Means clustering
def elbow_method_plot(values_norm_1991, values_norm_2001, axs):
    """
    Plot the elbow method to determine the optimal number of clusters.

    Parameters:
    - values_norm_1991 (array): Normalized data for the year 1991.
    - values_norm_2001 (array): Normalized data for the year 2001.
    - axs (Axes): The matplotlib axes to plot on.
    """
    y_1991 = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++',
                        max_iter=300, n_init=10, random_state=0)
        kmeans.fit(values_norm_1991)
        y_1991.append(kmeans.inertia_)

    y_2001 = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++',
                        max_iter=300, n_init=10, random_state=0)
        kmeans.fit(values_norm_2001)
        y_2001.append(kmeans.inertia_)

    axs.plot(range(1, 11), y_1991, marker='o',
             color='darkred', label='1991', linewidth=3.0)
    axs.plot(range(1, 11), y_2001, marker='o',
             color='goldenrod', label='2001', linewidth=3.0)
    axs.set_title('Elbow Method', fontweight='bold', fontsize=12)
    axs.set_xlabel('Number of clusters', fontweight='bold')
    axs.set_ylabel('Inertia (1991 & 2001)', fontweight='bold')
    axs.grid(True, linestyle='--', alpha=0.7)
    axs.legend()
    plt.savefig('elbow_curve.png')

# Function to process and plot GDP growth forecast for a specific country
def process_gdp_for_country(country_name, gdp_df, year_range, prediction_year):
    """
    Process and plot GDP growth forecast for a specific country.

    Parameters:
    - country_name (str): The name of the country.
    - gdp_df (DataFrame): GDP data as a pandas DataFrame.
    - year_range (array): Range of years for forecasting.
    - prediction_year (int): Year for which GDP growth is predicted.

    Returns:
    - prediction (float): Predicted GDP growth for the specified year.
    """
    country_gdp = gdp_df[['Year', country_name]].apply(
        pd.to_numeric, errors='coerce')
    param, cov = opt.curve_fit(
        polynomial, country_gdp['Year'], country_gdp[country_name])
    forecast = polynomial(year_range, *param)
    prediction = polynomial(np.array([prediction_year]), *param)
    prediction_label = f'Prediction for {prediction_year}: {prediction[0]:.2f}'

    error = get_error(country_gdp[country_name],
                      polynomial(country_gdp['Year'], *param), 2)
    upper_bound = forecast + error
    lower_bound = forecast - error

    plt.figure(figsize=(8, 6))
    plt.plot(country_gdp["Year"], country_gdp[country_name],
             label="GDP/Capita growth", c='dimgrey', linewidth=2.0)
    plt.plot(year_range, forecast, label="Forecast",
             c='darkred', linewidth=2.0)

    plt.fill_between(year_range, lower_bound, upper_bound,
                     color='goldenrod', alpha=0.3, label='Error range')

    plt.scatter(prediction_year, prediction, color='green',
                marker='o', label=prediction_label, s=100)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel("Year", fontweight='bold')
    plt.ylabel(
        "Total greenhouse gas emissions (kt of CO2 equivalent)",
        fontweight='bold')
    plt.legend()
    plt.title(f'{country_name} Total greenhouse gas emissions Forecast Prediction over Years',
              fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.7)
    # Save the plot with a filename based on the country name
    plt.savefig(f'{country_name}_plot.png')
    plt.show()
    plt.close()
    return prediction[0]

# Read and process greenhouse gas emissions data
dt_data, dt_transpose = process('API_19_DS2_en_excel_v2_6300761.xls', 3)

# Read and process GDP per capita growth data
gdp_data, gdp_transpose = process(
    'API_NY.GDP.PCAP.KD.ZG_DS2_en_excel_v2_6298376.xls', 3)

# Define columns and drop unnecessary columns
data_columns = ['Country Name', '1991', '2001']
drop_columns = ['Country Code', 'Indicator Code', 'Indicator Name']
indicator_name = 'Total greenhouse gas emissions (kt of CO2 equivalent)'

# Process greenhouse gas emissions data
green_gas = process_indicator_data(
    dt_data, indicator_name, data_columns, drop_columns)

# Process population data and normalize greenhouse gas emissions by population
per_pop = dt_data[dt_data['Indicator Name'] == 'Population, total']
per_pop = per_pop[data_columns]
per_pop = per_pop[per_pop['Country Name'].isin(
    green_gas.index)].set_index('Country Name')

years = ['1991', '2001']

for year in years:
    green_gas[year] = green_gas[year] / per_pop[year]

# Process GDP per capita growth data and filter relevant years
gdp_data = gdp_data[data_columns]
gdp_data = gdp_data[gdp_data['Country Name'].isin(green_gas.index)]
gdp_data = gdp_data.set_index('Country Name')

# Create dictionaries to store processed data for different years
dfs = {}

for year in years:
    df_year = green_gas.rename(columns={year: 'Total Green Gas emissions'})
    df_year['GDP per capita growth (annual %)'] = gdp_data[year]
    df_year = df_year.dropna(how='any', subset=[
                             'Total Green Gas emissions',
                             'GDP per capita growth (annual %)'])
    dfs[f'dt_{year}'] = df_year

dt_1991 = dfs['dt_1991']
dt_2001 = dfs['dt_2001']

# Create subplots for data plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot the data for the years 1991 and 2001
for i, year in enumerate(['1991', '2001']):
    plot_data(year, green_gas[year], gdp_data[year],
              axs[i], 'red' if year == '1991' else 'green')

# Create subplots for K-Means clustering
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Data for the years 1991 and 2001
data_years = {'1991': dt_1991, '2001': dt_2001}
normalized_values = {}

# Custom colors for clusters
custom_colors = ['red', 'green', 'yellow']  # Example custom colors

# Perform K-Means clustering for each year's data
for i, (year, data) in enumerate(data_years.items()):
    normalized_values[year], _ = perform_kmeans_clustering(
        data, year, axs[i], colors=custom_colors)

plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.7)

# Normalize the data for the years 1991 and 2001
value1_norm = normalize_data(dt_1991)
value2_norm = normalize_data(dt_2001)

# Create subplots for clustering with different numbers of clusters
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

predictions = {}

# Perform clustering with different numbers of clusters for 1991 and 2001
for i, year in enumerate(['1991', '2001']):
    values_norm = normalize_data(data_years[year])
    predictions[year] = n_clustering_and_plotting(values_norm, year, axs[i])

# Add cluster labels to the original dataframes
dt_1991['cluster'] = predictions['1991']
dt_2001['cluster'] = predictions['2001']

# Create a subplot for the elbow method plot
fig, axs = plt.subplots(figsize=(12, 5))
elbow_method_plot(value1_norm, value2_norm, axs)
plt.tight_layout()

# Process GDP data
gdp = gdp_transpose.drop(drop_columns)
gdp.columns = gdp.iloc[0, :]
gdp = gdp[1:]
gdp = gdp[gdp.columns[gdp.columns.isin(['India', 'Pakistan', 'China'])]]
gdp = gdp.dropna(how='any')
gdp['Year'] = gdp.index

india_gdp = gdp[['Year', 'India']].apply(pd.to_numeric, errors='coerce')
pakistan_gdp = gdp[['Year', 'Pakistan']].apply(pd.to_numeric, errors='coerce')
china_gdp = gdp[['Year', 'China']].apply(pd.to_numeric, errors='coerce')

# Prediction year for GDP forecasting
prediction_year = 2025

# Perform GDP forecasting and plotting for India, Pakistan, and China
predictions_2025 = {}
for country in ['India', 'Pakistan', 'China']:
    predictions_2025[country] = process_gdp_for_country(
        country, gdp, np.arange(1961, 2026), prediction_year)
