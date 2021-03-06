import dash
import numpy as np
from pathlib import Path
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from assignments.assignment1.a_load_file import read_dataset
from assignments.assignment1.e_experimentation import process_life_expectancy_dataset
from assignments.assignment1.b_data_profile import get_numeric_columns, get_text_categorical_columns
from assignments.assignment1.d_data_encoding import generate_label_encoder, replace_with_label_encoder
from assignments.assignment3.a_libraries import plotly_pie_chart
from assignments.assignment3.b_simple_usages import plotly_polar_scatterplot_chart
from assignments.assignment3.c_interactivity import plotly_interactivity


##############################################
# Now let's use dash, a library built on top of flask (a backend framework for python) and plotly
# Check the documentation at https://dash.plotly.com/
# For a complete example, check https://dash-bootstrap-components.opensource.faculty.ai/examples/iris/
# Example(s). Read the comments in the following method(s)
##############################################
def dash_simple_example():
    """
    Here is a simple example from https://dash.plotly.com/layout
    The focus is to create a fig from plotly and add it to dash, but differently from using just plotly, now we can use html elements,
    such as H1 for headers, Div for dividers, and all interations (buttons, sliders, etc).
    Check dash documentation for html and core components.
    """
    app = dash.Dash(__name__)

    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })

    # You create a fig just as you did in a_
    fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

    app.layout = html.Div(children=[
        html.H1(children='Hello Dash'),
        html.Div(children='Dash: A web application framework for Python.'),
        dcc.Graph(
            id='example-graph',
            figure=fig  # and include the fig here as a dcc.Graph
        )
    ])

    # To run the app, use app.run_server(debug=True)
    # However, do return the app instead so that the file can be run as in the end of this file
    return app


def dash_with_bootstrap_example():
    """
    Here is a simple example from https://dash.plotly.com/layout
    See examples of components from the bootstrap library at https://dash-bootstrap-components.opensource.faculty.ai/docs/components/alert/
    """
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })

    fig1 = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
    fig2 = px.line(df, x="Fruit", y="Amount", color="City")

    app.layout = dbc.Container([
        html.H1(children='Hello Dash'),
        html.Hr(),
        dbc.Row([
            dbc.Col(html.Div(children='Dash: A web application framework for Python.'), md=4),
            dbc.Col(dbc.Button('Example Button', color='primary', style={'margin-bottom': '1em'}, block=True), md=8)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='example-graph1', figure=fig1)),
            dbc.Col(dcc.Graph(id='example-graph2', figure=fig2))
        ])
    ])

    # To run the app, use app.run_server(debug=True)
    # However, do return the app instead so that the file can be run as in the end of this file
    return app


def dash_callback_example():
    """
    Here is a more complex example that uses callbacks. With this example, I believe you will suddenly perceive why dash (and webapps)
    are so much better for visual analysis.
    """
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })

    app.layout = dbc.Container([
        html.H1(children='Hello Dash'),
        html.Div(children='Dash: A web application framework for Python.'),
        html.Hr(),
        dbc.FormGroup([
            dbc.Label("Choose Dataset"),
            dcc.Dropdown(id="dropdown", value=1, options=[{"label": "First Data", "value": 1}, {"label": "Second Data", "value": 2}]),
        ]),
        dbc.FormGroup([
            dbc.Label(id='slider-value'),
            dcc.Slider(id="slider", min=1, max=10, step=0.5, value=1),
        ]),
        dbc.Button('Run Callback', id='example-button', color='primary', style={'margin-bottom': '1em'}, block=True),
        dbc.Row([
            dbc.Col(dcc.Graph(id='example-graph')),  # Not including fig here because it will be generated with the callback
        ])
    ])

    @app.callback(  # See documentation or tutorial to see how to use this
        Output('example-graph', 'figure'),  # Outputs is what you wish to update with the callback, which in this case is the figure
        [Input('example-button', 'n_clicks')],  # Use inputs to define when this callback is called, and read from the values in the inputs as parameters in the method
        [State('dropdown', 'value'),  # Use states to read values from the interface, but values only in states will not trigger the callback when changed
         State('slider', 'value')])  # For example, here if you change the slider, this method will not be called, it will only be called when you click the button
    def update_figure(n_clicks, dropdown_value, slider_value):
        df2 = df[:]
        df2.Amount = df2.Amount * slider_value
        if dropdown_value == 1:
            return px.bar(df2, x="Fruit", y="Amount", color="City", barmode="group")
        else:
            return px.line(df2, x="City", y="Amount", color="Fruit")

    @app.callback(Output('slider-value', 'children'), [Input('slider', 'value')])
    def update_slider_value(slider):
        return f'Multiplier: {slider}'

    #  You can also use app.callback to get selection from any of the plotly graphs, including tables and maps, and update anything you wish.
    #  See some examples at https://dash-gallery.plotly.host/Portal/

    return app


##############################################
# Implement all the below methods
# Use the documentation of the libraries in case you need help to understand the required methods
##############################################
def dash_task():
    """
    There is only only one task to do, a web app with:
    1. Some nice title
    2. One visualization placeholder for dataset visualization
        a. A dropdown to allow me to select which dataset I want to see (iris, video_game and life_expectancy)
        b. Two other dropdowns for me to choose what column to put in x and what column to put in y of the visualization
        c. Another dropdown for me to choose what type of graph I want (see examples in file a_) (at least 3 choices of graphs)
        d. Feel free to change the structure of the dataset if you prefer (e.g. change life_expectancy so that
            there is one column of "year", one for "country" and one for "value")
    4. A https://dash-bootstrap-components.opensource.faculty.ai/docs/components/card/ with the number of rows being showed on the above graph
    5. Another visualization with:
        a. It will containing the figure created in the tasks in a_, b_ or c_ related to plotly's figures
        b. Add a dropdown for me to choose among 3 (or more if you wish) different graphs from a_, b_ or c_ (choose the ones you like)
        c. In this visualization, if I select data in the visualization, update some text in the page (can be a new bootstrap card with text inside)
            with the number of values selected. (see https://dash.plotly.com/interactive-graphing for examples)
    """

    # Step-0 : Creating a new Dash Application Object with BootStrap properties
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    iris_df = read_dataset(Path("..", "..", "iris.csv"))
    iris_columns = list(iris_df.columns)

    video_game_df = read_dataset(Path("..", "..", "ratings_Video_Games.csv"))
    video_game_columns = list(video_game_df.columns)

    life_expectancy_df = process_life_expectancy_dataset()
    life_expectancy_columns = list(life_expectancy_df.columns)

    # Step-1 : Giving a Nice Title
    app.layout = dbc.Container([
        html.Center([
        html.H1(children='Dash Plots',style={'color' : 'brown'}),
        html.Hr(),

    ]),
        dbc.Row([
        dbc.Col(dbc.FormGroup([
            # Creating a dropdown for selecting the required dataset
            dbc.Label("Choose Your Dataset"),
            dcc.Dropdown(id='dataset_selector_dropdown',
                         options=[
                             {'label': 'Iris Dataset', 'value': 'iris'},
                             {'label': 'Video Game Dataset', 'value': 'video_game'},
                             {'label': 'Life Expectancy Dataset', 'value': 'life_expectancy'}
                         ],
                         value='iris',
                         style={'width': '250px'}

                         )
        ])),
            # Creating a dropdown for selecting X axis. Initially, it will be blank.
            # On Selecting a dataset from the aforementioned dropdown, the options will be
            # seen through the callback function : "setXandY()" declared below
        dbc.Col(dbc.FormGroup([
            dbc.Label("Select Column for X axis"),
            dcc.Dropdown(id='x_selector_dropdown', style={'width': '160px'})
        ])),
            # Creating a dropdown for selecting Y axis. Initially, it will be blank.
            # On Selecting a dataset from the aforementioned dropdown, the options will be
            # seen through the callback function : "setXandY()" declared below
        dbc.Col(dbc.FormGroup([
            dbc.Label("Select Column for Y axis"),
            dcc.Dropdown(id='y_selector_dropdown', style={'width': '160px'})
        ])),

            # Creating a dropdown for selecting the type of graph to be plotted
        dbc.Col(dbc.FormGroup([
            dbc.Label("Choose Graph Type"),
            dcc.Dropdown(id='graph_type_selector_dropdown',
                             options=[
                                 {'label': 'Scatter Plot', 'value': 'scatter'},
                                 {'label': 'Histogram Plot', 'value': 'histogram'},
                                 {'label': 'Polar Plot', 'value': 'polar'}
                             ],
                             value='scatter',
                             style={'width': '160px'}

                        )
            ])),
        ]),
        # Creating a button on click of which the Dash program will display the required plot
        dbc.Row(dbc.Button('Generate Plot',id='button_for_generating_plot', style={'width':'100%'}, color='primary')),

        # Creating the first graph component
        html.Center([
            dbc.Col(dcc.Graph(id='first_plot'))
        ]),

        # Creating the card component for counting the scanned rows
        html.Center([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([ html.H4(children=['Number of rows read : 0'], id='card_text')])
                   ], style={"width": "18rem"})
            )
        ]),
        html.Br(),
        # Creating a dropdown for selecting one visualization out of 3 from a_, b_ or c_ files
            dbc.FormGroup([
                dbc.Label("Choose a Visualization"),
                dcc.Dropdown(id='second_plot_selector_dropdown',
                             options=[
                                 {'label': 'Pie Chart From File A', 'value': 'a_pie'},
                                 {'label': 'Polar Scatter Plot From File B', 'value': 'b_polar'},
                                 {'label': 'Cluster Plot From File C', 'value': 'c_clusters'}
                             ],
                             value='c_clusters',
                             style={'width': '100%'}
                             )
            ]),
            html.Br(),
            html.Center([dcc.Graph(id='second_plot')]),

        # Creating another Card for viewing the selected data from the second plot
            html.Center([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([html.H4(children=['No Data Selected'], id='second_card_text')])
                ], style={"width": "18rem"})
            )
        ])
   ])

    # The callback function below sets the X and Y dropdown options on selection of dataset selection
    @app.callback(
            [
                Output('x_selector_dropdown','options'),
                Output('x_selector_dropdown', 'value'),
                Output('y_selector_dropdown', 'options'),
                Output('y_selector_dropdown', 'value')
            ],
            [Input('dataset_selector_dropdown', 'value')])
    def setXandY(dropdownValue):

        required_options = []
        default_value = ''
        if dropdownValue == 'iris':
            for data_column in iris_columns:
                temp_dict = {}
                temp_dict['label'] = data_column
                temp_dict['value'] = data_column
                required_options.append(temp_dict)

            default_value = iris_columns[0]

        elif dropdownValue == 'video_game':
            for data_column in video_game_columns:
                temp_dict = {}
                temp_dict['label'] = data_column
                temp_dict['value'] = data_column
                required_options.append(temp_dict)

            default_value = video_game_columns[0]

        elif dropdownValue == 'life_expectancy':
            for data_column in life_expectancy_columns:
                temp_dict = {}
                temp_dict['label'] = data_column
                temp_dict['value'] = data_column
                required_options.append(temp_dict)

            default_value = life_expectancy_columns[0]

        return required_options, default_value, required_options, default_value

    # Callback function for plotting the first plot and updating the first card
    @app.callback( [
        Output('first_plot', 'figure'),
        Output('card_text', 'children')
        ],
        [
            Input('button_for_generating_plot', 'n_clicks'),
        ],
        [
            State('dataset_selector_dropdown', 'value'),
            State('x_selector_dropdown', 'value'),
            State('y_selector_dropdown', 'value'),
            State('graph_type_selector_dropdown', 'value')
        ]
    )

    def plot_first_graph(n_clicks, dataset_name, x_column, y_column, graph_type):

        if n_clicks==None:
            return go.Figure(),'Rows Scanned : 0'

        if dataset_name == 'iris':
            df = iris_df
            rows_count = len(iris_df)
        elif dataset_name == 'video_game':
            df = video_game_df
            rows_count = len(video_game_df)
        elif dataset_name == 'life_expectancy':
            df = life_expectancy_df
            rows_count = len(life_expectancy_df)
        else:
            df = None



        categorical_cols = get_text_categorical_columns(df)

        if x_column in categorical_cols:
            le = generate_label_encoder(df[x_column])
            df = replace_with_label_encoder(df, x_column, le)

        if y_column in categorical_cols:
            le = generate_label_encoder(df[y_column])
            df = replace_with_label_encoder(df, y_column, le)

        if graph_type == 'scatter':
            first_figure = px.scatter(df, x=x_column, y=y_column)

        elif graph_type == 'histogram':
            first_figure = px.histogram(df, x=x_column, color=y_column)

        elif graph_type == 'polar':
            first_figure = px.scatter_polar(df, r=x_column, theta=y_column)

        else:
            first_figure = None

        final_rows_call = 'Rows Read : ' + str(rows_count)

        return first_figure, final_rows_call

    # Callback function for plotting the second plot
    @app.callback(
        Output('second_plot','figure'),
        [Input('second_plot_selector_dropdown', 'value')]
    )
    def generate_second_plot(plot_name):

        if plot_name == 'a_pie':
            x = np.random.rand(50) * np.random.randint(-10, 10)
            y = np.random.rand(50) * np.random.randint(-10, 10)
            df = pd.DataFrame(dict(x=x, y=y, z=x + y))
            fig = plotly_pie_chart(df)

        elif plot_name == 'b_polar':
            fig = plotly_polar_scatterplot_chart()

        elif plot_name == 'c_clusters':
            fig = plotly_interactivity()

        else:
            fig = None

        return fig

    # Callback function for updating the second card. Please Select Lasso Select and then perform selection via dragging
    @app.callback(
            Output('second_card_text','children'),
            Input('second_plot','selectedData')
    )
    def act_on_selecting_data(selectedData):
        print(json.dumps(selectedData, indent=2))
        print(type(selectedData))
        return json.dumps(selectedData, indent=2)

    return app

if __name__ == "__main__":
    # Here are examples of how to run each method
    # Although you may remove or comment them out for testing, make sure they are present
    # and work when submitting, since when we run your code, all methods will be run like this.
    # If these lines below returns errors when running, your file will be considered to not
    # run, and graded accordingly.
    app_ce = dash_callback_example()
    app_b = dash_with_bootstrap_example()
    app_c = dash_callback_example()
    app_t = dash_task()

    # Uncomment the below lines to test your code
    # When submitting, leave the code below commented!!!
    #app_ce.run_server(debug=True)
    #app_b.run_server(debug=True)
    #app_c.run_server(debug=True)
    #app_t.run_server(debug=True)
