import math
from typing import Tuple
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.widgets import Button, Slider
from assignments.assignment3.b_simple_usages import *
from assignments.assignment1.e_experimentation import process_iris_dataset_again


###############
# Interactivity in visualizations is challenging due to limitations and clunkiness of libraries.
# For example, some libraries works well in Jupyter Notebooks, but then the code makes barely any sense and becomes hard to change/update,
# defeating the purpose of using Jupyter notebooks in the first place, and other libraries provide a window of their own, but
# they are very tied to the running code, and far from the experience you'd expect from a proper visual analytics webapp
#
# We will try out some approaches to exercise in this file, but the next file will give you the proper tooling to make a
# well-rounded and efficient code for visual interactions.
##############################################
# Example(s). Read the comments in the following method(s)
##############################################


def matplotlib_simple_example():
    """
    Using the same logic from before, we can add sliders or buttons to select options for interactivity.
    Matplotlib is limited to only a few options, but do take a look, since they are useful for fast prototyping and analysis

    In case you are using PyCharm, I suggest you to uncheck the 'Show plots in tool window'
    to be able to see and interact with the buttons defined below.
    This example comes from https://matplotlib.org/3.1.1/gallery/widgets/buttons.html
    """
    freqs = np.arange(2, 20, 3)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    t = np.arange(0.0, 1.0, 0.001)
    s = np.sin(2 * np.pi * freqs[0] * t)
    l, = plt.plot(t, s, lw=2)

    class Index(object):
        ind = 0

        def next(self, event):
            self.ind += 1
            i = self.ind % len(freqs)
            ydata = np.sin(2 * np.pi * freqs[i] * t)
            l.set_ydata(ydata)
            plt.show()

        def prev(self, event):
            self.ind -= 1
            i = self.ind % len(freqs)
            ydata = np.sin(2 * np.pi * freqs[i] * t)
            l.set_ydata(ydata)
            plt.show()

    callback = Index()
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)

    plt.show()
    return fig, ax


def matplotlib_simple_example2():
    """
    Here is another example, which also has a slider and simplifies a bit the callbacks
    """
    data = np.random.rand(10, 5)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    ax.bar(np.arange(10).astype(str).tolist(), data[:, 0])

    class Index(object):
        ind = 0
        multiplier = 1

        def change_data(self, event, i):
            self.ind = np.clip(self.ind + i, 0, data.shape[1] - 1)
            ax.clear()
            ax.bar(np.arange(10).astype(str).tolist(),
                   data[:, self.ind] * self.multiplier)
            plt.show()

        def change_multiplier(self, value):
            self.multiplier = value
            ax.clear()
            ax.bar(np.arange(10).astype(str).tolist(),
                   data[:, self.ind] * self.multiplier)
            plt.show()

    callback = Index()
    axprev = plt.axes([0.1, 0.05, 0.12, 0.075])
    axnext = plt.axes([0.23, 0.05, 0.12, 0.075])
    axslider = plt.axes([0.55, 0.1, 0.35, 0.03])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(lambda event: callback.change_data(event, 1))
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(lambda event: callback.change_data(event, -1))
    slider = Slider(axslider, 'my dfd  multiplier', 1, 10, 1)
    slider.on_changed(callback.change_multiplier)
    plt.show()
    return fig, ax


def plotly_slider_example():
    """
    Here is a simple example from https://plotly.com/python/sliders/ of how to include a slider in plotly
    Notice how similar it already is to GapMinder!
    """
    df = px.data.gapminder()
    fig = px.scatter(df, x="gdpPercap", y="lifeExp",
                     animation_frame="year",  # set which column makes the animation though a slider
                     size="pop",
                     color="continent",
                     hover_name="country",
                     log_x=True,
                     size_max=55,
                     range_x=[100, 100000],
                     range_y=[25, 90])

    fig["layout"].pop("updatemenus")  # optional, drop animation buttons

    return fig


def plotly_button_example():
    """
    To have buttons, plotly requires us to use go (and not px) to generate the graph.
    The button options are very restrictive, since all they can do is change a parameter from the go graph.
    In the example below, it changes the "mode" value of the graph (between lines and scatter)
    The code is a modified code example taken from https://plotly.com/python/custom-buttons/
    """
    x = np.random.rand(50) * np.random.randint(-10, 10)
    y = np.random.rand(50) * np.random.randint(-10, 10)

    fig = go.Figure()

    # Add surface trace
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers'))

    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(type="buttons",
                 direction="left",
                 buttons=[
                     dict(
                         label="line",  # just the name of the button
                         # This is the method of update (check https://plotly.com/python/custom-buttons/)
                         method="update",
                         # This is the value being updated in the visualization
                         args=[{"mode": "markers"}],
                     ), dict(
                         label="scatter",  # just the name of the button
                         # This is the method of update (check https://plotly.com/python/custom-buttons/)
                         method="update",
                         # This is the value being updated in the visualization
                         args=[{"mode": "line"}],
                     )
                 ],
                 # Layout-related values
                 pad={"r": 10, "t": 10}, showactive=True, x=0.11, xanchor="left", y=1.1, yanchor="top"
                 ),
        ]
    )

    fig.show()
    return fig


##############################################
# Implement all the below methods
# Use the documentation of the libraries in case you need help to understand the required methods
##############################################
def matplotlib_interactivity():
    """
    Do an interactive matplotlib plot where I can select which visualization I want.
    Make either a slider, a dropdown or several buttons and make so each option gives me a different visualization from
    the matplotlib figures of b_simple_usages. Return just the resulting fig as is done in plotly_slider_example.
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
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

    iris_data = read_dataset(Path('..', '..', 'iris.csv'))
    Numerical_Columns = get_numeric_columns(iris_data)

    Max_Values = []

    for data_column in Numerical_Columns:
        temp = get_column_max(iris_data, data_column)
        Max_Values.append(temp)

    Label_Values = ['sepal_length', 'sepal_width',
                    'petal_length', 'petal_width']

    df = process_life_expectancy_dataset()
    Numerical_Columns = get_numeric_columns(df)
    Categorical_Columns = get_text_categorical_columns(df)
    Binary_Columns = get_binary_columns(df)

    N_Count = len(Numerical_Columns)
    C_Count = len(Categorical_Columns)
    B_Count = len(Binary_Columns)

    column_type_data = [N_Count, C_Count, B_Count]
    data_labels = ['Numeric Columns', 'Categorical Columns', 'Binary Columns']

    class MyFigure:

        def show_bar_plot(self, event):

            ax[0,0].clear()
            ax[0, 1].clear()
            ax[1, 0].clear()
            ax[1, 1].clear()

            ax[0,0].bar(Label_Values, Max_Values)
            plt.show()

        def show_pie_plot(self, event):

            ax[0, 0].clear()
            ax[0, 1].clear()
            ax[1, 0].clear()
            ax[1, 1].clear()


            ax[0,0].pie(column_type_data, labels=data_labels)
            plt.show()

        def show_hist(self, event):

            ax[0, 0].clear()
            ax[0, 1].clear()
            ax[1, 0].clear()
            ax[1, 1].clear()
            df = read_dataset(Path('..', '..', 'iris.csv'))
            sepal_lengths = np.array(df.sepal_length)
            sepal_widths = np.array(df.sepal_width)
            petal_lengths = np.array(df.petal_length)
            petal_widths = np.array(df.petal_width)


            ax[0, 0].hist(sepal_lengths, bins=4)
            ax[0, 1].hist(sepal_widths, bins=4)
            ax[1, 0].hist(petal_lengths, bins=4)
            ax[1, 1].hist(petal_widths, bins=4)

            plt.show()

        def show_heat_map(self, event):

            ax[0, 0].clear()
            ax[0, 1].clear()
            ax[1, 0].clear()
            ax[1, 1].clear()

            ax[0,0].imshow(corr_Data, cmap='hot', interpolation='nearest')
            plt.show()

    iris_data = read_dataset(Path('..', '..', 'iris.csv'))
    Numerical_Columns = get_numeric_columns(iris_data)

    Max_Values = []

    for data_column in Numerical_Columns:
        temp = get_column_max(iris_data, data_column)
        Max_Values.append(temp)

    Label_Values = ['sepal_length', 'sepal_width',
                    'petal_length', 'petal_width']

    fig, ax = plt.subplots(2,2)
    ax[0,0].bar(Label_Values, Max_Values)
    plt.subplots_adjust(bottom=0.25)
    my_figure = MyFigure()
    ax1 = plt.axes([0.20, 0.01, 0.2, 0.05])
    ax2 = plt.axes([0.40, 0.01, 0.2, 0.05])
    ax3 = plt.axes([0.60, 0.01, 0.2, 0.05])
    ax4 = plt.axes([0.80, 0.01, 0.2, 0.05])

    b1 = Button(ax1, "Bar_Plot")
    b1.on_clicked(my_figure.show_bar_plot)

    b2 = Button(ax2, "Pie_Plot")
    b2.on_clicked(my_figure.show_pie_plot)

    b3 = Button(ax3, "Histogram")
    b3.on_clicked(my_figure.show_hist)

    b4 = Button(ax4, "Heat_Map")
    b4.on_clicked(my_figure.show_heat_map)


    plt.show()
    return fig, ax

# Helper method for performing clustering
def perform_clustering(number_of_clusters):

    df = process_iris_dataset_again()
    le = generate_label_encoder(df['large_sepal_length'])
    df = replace_with_label_encoder(df, 'large_sepal_length', le)

    model = KMeans(n_clusters=number_of_clusters)
    clusters = model.fit_transform(df)

    score = metrics.silhouette_score(df, model.labels_, metric='euclidean')
    data_dict = dict(model=model, score=score, clusters=clusters)


    return data_dict

def matplotlib_cluster_interactivity():
    """
    Do an interactive matplotlib plot where I can select how many clusters I want to train from.
    Use iris dataset (just numeric columns) and k-means (feel free to reuse as/c_clustering if you wish).
    The slider (or dropdown) should range from 2 to 10. Return just the resulting fig.
    """

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    class ClusterPlot:

        def cluster_change(self, new_cluster_count):

            data_dict = perform_clustering(round(new_cluster_count))

            clusters = data_dict['clusters']
            two_d_clusters = np.delete(clusters, 1, 1)
            c_centers = data_dict['model'].cluster_centers_


            ax.clear()
            print(two_d_clusters[:,0])
            ax.scatter(x=two_d_clusters[:, 0],
                y=two_d_clusters[:, 1],
                c=data_dict['model'].labels_)

            plt.show()

    my_cluster_plot = ClusterPlot()
    axslider = plt.axes([0.60, 0.01, 0.2, 0.05])
    slider = Slider(axslider, 'Set K', 2,10,valfmt="%0.0f")
    slider.on_changed(my_cluster_plot.cluster_change)
    plt.show()
    return None


def plotly_interactivity():
    """
    Do a plotly graph with all plotly 6 figs from b_simple_usages, and make 6 buttons (one for each fig).
    Change the displayed graph depending on which button I click. Return just the resulting fig.
    """

    # Getting the required plots
    fig_p_s = plotly_scatter_plot_chart()
    fig_p_bpc = plotly_bar_plot_chart()
    fig_p_psc = plotly_polar_scatterplot_chart()
    fig_p_t = plotly_table()
    fig_p_clb = plotly_composite_line_bar()
    fig_p_map = plotly_map()

    # Converting the data into the required format asked by plotly for plotting on click of Buttons
    scatter_plot_dict = {key: [value] for key, value in fig_p_s.to_dict()['data'][0].items()}
    bar_plot_dict = {key: [value] for key, value in fig_p_bpc.to_dict()['data'][0].items()}
    polar_plot_dict = {key: [value] for key, value in fig_p_psc.to_dict()['data'][0].items()}
    table_plot_dict = {key: [value] for key, value in fig_p_t.to_dict()['data'][0].items()}
    line_bar_plot_dict ={key: [value] for key, value in fig_p_clb.to_dict()['data'][0].items()}
    map_plot_dict = {key: [value] for key, value in fig_p_map.to_dict()['data'][0].items()}

    fig = copy.copy(fig_p_s)
    # Setting up the logic of generating plots on Button Clicks
    fig.update_layout(
        updatemenus=[
            dict(type="buttons",
                 direction="left",
                 buttons=[
                     dict(
                         label="Scatter Plot",

                         method="update",

                         args=[scatter_plot_dict],
                     ), dict(
                         label="Bar Plot",

                         method="update",

                         args=[bar_plot_dict],
                     ), dict(
                         label="Polar Plot",

                         method="update",

                         args=[polar_plot_dict],
                     ), dict(
                         label="Table Plot",

                         method="update",

                         args=[table_plot_dict],
                     ), dict(
                         label="Map Plot",

                         method="update",

                         args=[map_plot_dict],
                     ), dict(
                         label="Line Bar Plot",

                         method="update",

                         args=[line_bar_plot_dict],
                     )
                 ],

                 pad={"r": 10, "t": 10}, showactive=True, x=0.11, xanchor="left", y=1.1, yanchor="top"
                 ),
        ]
    )
    return fig


if __name__ == "__main__":
    # Here are examples of how to run each method
    # Although you may remove or comment them out for testing, make sure they are present
    # and work when submitting, since when we run your code, all methods will be run like this.
    # If these lines below returns errors when running, your file will be considered to not
    # run, and graded accordingly.
    fig_m_i = matplotlib_interactivity()
    fig_m_ci = matplotlib_cluster_interactivity()
    fig_p = plotly_interactivity()

    # Uncomment the below lines to test your code
    # When submitting, leave the code below commented!!!
    # matplotlib_simple_example()[0].show()
    #matplotlib_simple_example2()[0].show()
    # plotly_slider_example().show()
    #plotly_button_example().show()
    # fig_m_i.show()
    # fig_m_ci.show()
    # fig_p.show()
