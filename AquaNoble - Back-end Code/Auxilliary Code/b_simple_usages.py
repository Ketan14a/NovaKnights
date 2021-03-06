from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from assignments.assignment1.c_data_cleaning import *
from assignments.assignment1.a_load_file import *
from assignments.assignment1.b_data_profile import *
from assignments.assignment1.d_data_encoding import *
from assignments.assignment1.e_experimentation import *
from assignments.assignment2.a_classification import your_choice
from assignments.assignment2.a_classification import *
from assignments.assignment2.c_clustering import *
from assignments.assignment2.d_extra import *
from assignments.assignment3.a_libraries import *

##############################################
# In this file, we will use data and methods of previous assignments with visualization.
# But before you continue on, take some time to look on the internet about the many existing visualization types and their usages, for example:
# https://extremepresentation.typepad.com/blog/2006/09/choosing_a_good.html
# https://datavizcatalogue.com/
# https://plotly.com/python/
# https://www.tableau.com/learn/whitepapers/which-chart-or-graph-is-right-for-you
# Or just google "which visualization to use", and you'll find a near-infinite number of resources
#
# You may want to create a new visualization in the future, and for that I suggest using JavaScript and D3.js, but for the course, we will only
# use python and already available visualizations
##############################################
# Implement all the below methods
# Use the documentation of the libraries in case you need help to understand the required methods
# For ALL methods return the fig and ax of matplotlib or fig from plotly!
##############################################


def matplotlib_bar_chart() -> Tuple:
    """
    Create a bar chart with a1/b_data_profile's get column max.
    Show the max of each numeric column from iris dataset as the bars
    """

    # Reading the required dataset and getting the dataframe
    iris_data = read_dataset(Path('..', '..', 'iris.csv'))

    # Identifying the numerical columns from the iris dataset
    Numerical_Columns = get_numeric_columns(iris_data)

    Max_Values = []

    # Getting the required data in X and Y shape
    for data_column in Numerical_Columns:
        temp = get_column_max(iris_data, data_column)
        Max_Values.append(temp)

    Label_Values = ['sepal_length', 'sepal_width',
                    'petal_length', 'petal_width']

    # Plotting the transformed data
    fig, ax = plt.subplots()
    ax.bar(Label_Values, Max_Values)
    return fig, ax


def matplotlib_pie_chart() -> Tuple:
    """
    Create a pie chart where each piece of the chart has the number of columns which are numeric/categorical/binary
    from the output of a1/e_/process_life_expectancy_dataset
    """

    # Reading the required dataframe
    df = process_life_expectancy_dataset()

    # Identifying the column types
    Numerical_Columns = get_numeric_columns(df)
    Categorical_Columns = get_text_categorical_columns(df)
    Binary_Columns = get_binary_columns(df)

    # Counting each type
    N_Count = len(Numerical_Columns)
    C_Count = len(Categorical_Columns)
    B_Count = len(Binary_Columns)

    # Getting the data into required shape for visualization
    column_type_data = [N_Count, C_Count, B_Count]
    data_labels = ['Numeric Columns', 'Categorical Columns', 'Binary Columns']

    # Plotting the data
    fig, ax = plt.subplots()
    ax.pie(column_type_data, labels=data_labels)

    return fig, ax


def matplotlib_histogram() -> Tuple:
    """
    Build 4 histograms as subplots in one figure with the numeric values of the iris dataset
    """

    # Reading the required dataframe
    df = read_dataset(Path('..', '..', 'iris.csv'))

    # Transforming the data into required shape
    sepal_lengths = np.array(df.sepal_length)
    sepal_widths = np.array(df.sepal_width)
    petal_lengths = np.array(df.petal_length)
    petal_widths = np.array(df.petal_width)

    # Plotting the histograms
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].hist(sepal_lengths, bins=4)
    ax[0, 1].hist(sepal_widths, bins=4)
    ax[1, 0].hist(petal_lengths, bins=4)
    ax[1, 1].hist(petal_widths, bins=4)

    return fig, ax


def matplotlib_heatmap_chart() -> Tuple:
    """
    Use the pearson correlation (e.g. https://docs.scipy.org/doc/scipy-1.5.3/reference/generated/scipy.stats.pearsonr.html)
    Remember a1/b_/pandas_profile? There is a heat map over there to analyse the correlation among columns.
    to calculate the correlation between two numeric columns and show that as a heat map. Use the iris dataset.
    """

    # Reading the required dataset
    df = read_dataset(Path('..', '..', 'iris.csv'))

    # Getting the required data into shape
    data_columns = ['sepal_length', 'sepal_width',
                    'petal_length', 'petal_width']
    Pearsons_coefficients = []
    for i in data_columns:
        temp = []
        for j in data_columns:
            pc = get_correlation_between_columns(df, i, j)
            temp.append(pc)

        Pearsons_coefficients.append(temp)

    corr_Data = np.array(Pearsons_coefficients)

    # Plotting the data
    fig, ax = plt.subplots()
    ax.imshow(corr_Data, cmap='hot', interpolation='nearest')
    return fig, ax


# There are many other possibilities. Please, do check the documentation and examples so you
# may have a good breadth of tools for future work (in assignments, projects, and your own career)
###################################
# Once again, for ALL methods return the fig and ax of matplotlib or fig from plotly!


def plotly_scatter_plot_chart():
    """
    Use the result of a2/c_clustering/cluster_iris_dataset_again() as the color of a scatterplot made from the original (unprocessed)
    iris dataset. Choose among the numeric values to be the x and y coordinates.
    """

    # Reading the clusters into this code
    data_dict = cluster_iris_dataset_again()

    # Accessing the clusters
    clusters = data_dict['clusters']

    # Cutting the cluster array as per the plot's requirement
    two_d_clusters = np.delete(clusters, 1, 1)

    # Accessing the cluster centers
    c_centers = data_dict['model'].cluster_centers_
    two_d_centers = np.delete(c_centers, 1, 1)

    two_d_centers = np.delete(two_d_centers, 2, 1)
    two_d_centers = np.delete(two_d_centers, 3, 1)
    two_d_centers = np.delete(two_d_centers, 0, 1)

    # Plotting the clusters
    fig = go.Figure(data=go.Scatter(
        x=two_d_clusters[:, 0],
        y=two_d_clusters[:, 1],
        mode='markers',
        marker=dict(
            size=16,
            # setting color equal to cluster labels
            color=data_dict['model'].labels_,
            colorscale='Viridis',  # one of plotly colorscales
            showscale=True
        )
    ))

    return fig


def plotly_bar_plot_chart():
    """
    Use the result of a2/c_clustering/cluster_iris_dataset_again() and use x as 3 groups of bars (one for each iris species)
    and each group has multiple bars, one for each cluster, with y as the count of instances in the specific cluster/species combination.
    # grouped-bar-chart (search for the grouped bar chart visualization)
    The grouped bar chart is like https://plotly.com/python/bar-charts/
    """

    # Reading the required clusters into a dictionary
    data_dict = cluster_iris_dataset_again()
    clusters = data_dict['model'].labels_

    # Getting the counts of cluster points in each cluster
    setosa_count = np.count_nonzero(clusters == 0)
    versicolor_count = np.count_nonzero(clusters == 1)
    virginica_count = np.count_nonzero(clusters == 2)

    # plotting the clusters as bar graphs
    fig = go.Figure(data=[
        go.Bar(name='Setosa', x=['setosa'], y=[setosa_count]),
        go.Bar(name='Versicolor', x=['versicolor'], y=[versicolor_count]),
        go.Bar(name='Virginica', x=['virginica'], y=[virginica_count])
    ])

    # For plotting "grouped" bar graphs
    fig.update_layout(barmode='group')

    return fig


def plotly_polar_scatterplot_chart():
    """
    Do something similar to a1/e_/process_life_expectancy_dataset, but don't drop the latitude and longitude.
    Use these two values to figure out the theta to plot values as a compass (example: https://plotly.com/python/polar-chart/).
    Each point should be one country and the radius should be thd value from the dataset (add up all years and feel free to ignore everything else)
    """

    # Reading the required dataframe
    le_df = read_dataset(Path('..', '..', 'life_expectancy_years.csv'))

    # Processing the dataframe for fitting it into visualization
    le_df = le_df.dropna()
    le_df['nations'] = le_df['country']
    temp_df = le_df.loc[:, le_df.columns != 'country']

    le_df['total'] = temp_df.sum(axis=1)
    le_df['nations'] = le_df['country']
    ge_df = read_dataset(Path('..', '..', 'geography.csv'))

    final_df = pd.concat(
        [le_df.set_index('country'), ge_df.set_index('name')], axis=1, join='inner')
    final_df = final_df[['nations', 'Latitude', 'Longitude', 'total']]

    # Plotting the data
    fig = px.scatter_polar(final_df, r="total", theta="Longitude")

    return fig


def plotly_table():
    """
    Show the data from a2/a_classification/your_choice() as a table
    See https://plotly.com/python/table/ for documentation
    """

    # Reading the data
    my_data = your_choice()

    # Generating the table columns
    c1 = ['model', 'Accuracy', 'Test_Prediction']
    c2 = [str(my_data['model']), str(my_data['accuracy']),
          str(my_data['test_prediction'])]

    # Plotting the table
    fig = go.Figure(data=[go.Table(header=dict(values=['Result Parameters', 'Result Values']),
                                   cells=dict(values=[c1, c2]))
                          ])

    return fig


def plotly_composite_line_bar():
    """
    Use the data from a1/e_/process_life_expectancy_dataset and show in a single graph on year on x and value on y where
    there are 5 line charts of 5 countries (you choose which) and one bar chart on the background with the total value of all 5
    countries added up.
    """
    le_df = read_dataset(Path('..', '..', 'life_expectancy_years.csv'))
    le_df = le_df.dropna()
    le_df['nations'] = le_df['country']
    temp_df = le_df.loc[:, le_df.columns != 'country']

    le_df['total'] = temp_df.sum(axis=1)
    le_df['nations'] = le_df['country']
    ge_df = read_dataset(Path('..', '..', 'geography.csv'))

    final_df = pd.concat(
        [le_df.set_index('country'), ge_df.set_index('name')], axis=1, join='inner')

    sliced_df = final_df.iloc[20:25, :]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=sliced_df['nations'],
            y=sliced_df['total']
        ))

    fig.add_trace(
        go.Bar(
            x=sliced_df['nations'],
            y=sliced_df['total']
        ))

    return fig


def plotly_map():
    """
    Use the data from a1/e_/process_life_expectancy_dataset on a plotly map (anyone will do)
    # using-builtin-country-and-state-geometries
    Examples: https://plotly.com/python/maps/, https://plotly.com/python/choropleth-maps/
    Use the value from the dataset of a specific year (e.g. 1900) to show as the color in the map
    """
    le_df = read_dataset(Path('..', '..', 'life_expectancy_years.csv'))
    le_df = le_df.dropna()
    le_df['nations'] = le_df['country']
    temp_df = le_df.loc[:, le_df.columns != 'country']

    le_df['total'] = temp_df.sum(axis=1)
    le_df['nations'] = le_df['country']
    ge_df = read_dataset(Path('..', '..', 'geography.csv'))

    final_df = pd.concat(
        [le_df.set_index('country'), ge_df.set_index('name')], axis=1, join='inner')

    final_df = final_df[['nations', '1900', 'Latitude', 'Longitude']]

    data = dict(
        type='choropleth',
        locations=final_df['nations'],
        locationmode='country names',
        colorscale='gnbu',
        z=final_df['1900'])

    fig = go.Figure(data=[data])

    return fig


def plotly_tree_map():
    """
    Use plotly's treemap to plot any data returned from any of a1/e_experimentation or a2 tasks
    Documentation: https://plotly.com/python/treemaps/
    """
    df = process_iris_dataset()
    fig = px.treemap(
        df, path=['x0_setosa', 'x0_versicolor', 'x0_virginica'], values='numeric_mean')

    return fig


if __name__ == "__main__":
    # Here are examples of how to run each method
    # Although you may remove or comment them out for testing, make sure they are present
    # and work when submitting, since when we run your code, all methods will be run like this.
    # If these lines below returns errors when running, your file will be considered to not
    # run, and graded accordingly.
    fig_m_bc, _ = matplotlib_bar_chart()
    fig_m_pc, _ = matplotlib_pie_chart()
    fig_m_h, _ = matplotlib_histogram()
    fig_m_hc, _ = matplotlib_heatmap_chart()

    fig_p_s = plotly_scatter_plot_chart()
    fig_p_bpc = plotly_bar_plot_chart()
    fig_p_psc = plotly_polar_scatterplot_chart()
    fig_p_t = plotly_table()
    fig_p_clb = plotly_composite_line_bar()
    fig_p_map = plotly_map()
    # Uncomment the below lines to test your code
    # When submitting, leave the code below commented!!!
    # fig_m_bc.show()
    # fig_m_pc.show()
    # fig_m_h.show()
    # fig_m_hc.show()
    #
    # fig_p_s.show()
    # fig_p_bpc.show()
    # fig_p_psc.show()
    # fig_p_t.show()
    # fig_p_clb.show()
    # fig_p_map.show()
    # fig_p_treemap.show()
