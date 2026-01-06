import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import os

os.chdir('C:\\Dania\\2024\\Australian Election')

# Load your data
df = pd.read_csv("PartyVotePercentByDivision2022.csv") 

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("2022 Federal Election Results By Division"),
    dash_table.DataTable(
        id='election-table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        sort_action='native',
        filter_action='native',
        page_action='native',
        page_size=160,  # number of rows per page
        style_data_conditional=[
            {
                'if': {'column_id': 'COAL'},
                'backgroundColor': '#D3D6FE',  # Blue
                'color': 'black',            },
            {
                'if': {'column_id': 'ALP'},
                'backgroundColor': '#FED4D3',  # Red
                'color': 'black',
            },
            {
                'if': {'column_id': 'GRN'},
                'backgroundColor': '#D4FED3',  # Grn
                'color': 'black',
            },
            {
                'if': {'column_id': 'IND'},
                'backgroundColor': '#E3E3E3',  # Grey
            },
            {
                'if': {'column_id': 'ON'},
                'backgroundColor': '#FEE9DA',  # Red
            },
            {
                'if': {'column_id': 'UAPP'},
                'backgroundColor': '#FBFED3',  # Red
                'color': 'black',
            },
            {
                'if': {'column_id': 'LDP'},
                'backgroundColor': '#F3FED3',  # Red
            },
            {
                'if': {'column_id': 'AJP'},
                'backgroundColor': '#EFD9FE',  # Purp
            },
            {
                'if': {'column_id': 'XEN'},
                'backgroundColor': '#FEE2D3',  # Red
            },
            {
                'if': {'column_id': 'KAP'},
                'backgroundColor': '#F4EBE8',  # Brown
            }]
    )
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)