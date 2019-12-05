from pathlib import Path
import os
import sys
source_path = str(Path(os.path.abspath(__file__)).parent.parent)
production_path = str(Path(os.path.abspath(__file__)).parent.parent / 'production')
sys.path = [i for i in sys.path if i != production_path]

if source_path not in sys.path:
    sys.path.insert(0, source_path)

import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
from dash.dependencies import Input, Output, State

from production.production import generate_prioritized_risk_list


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

risk_factors = ['value on agency', 'bidders on agency', 'value on tender', 'value on type of procurement',
                'complaints on type of procurement']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([

    html.H1("DNCP", style={'color': 'orange', 'fontWeight': 'bold'}),

    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Prioritized tenders', children=[
            html.Div(
                [html.Div("Input file name with the new tenders to review:"),
                dcc.Input(id='file-name', type='text'),

                html.Br([])],
                className="two columns offset-by-one"
            ),

            html.Div([
                html.Div([
                    html.Button(id='submit-button',
                            children='Submit'
                    )],
                className="two columns offset-by-one")
            ]),

            html.Div(
                [html.Br([]),
                html.H3("Prioritized list of tenders",
                        style={'fontWeight': 'bold'}),

                html.Div(id="priority-table")],
                className="eleven columns"
            )
        ]),

        dcc.Tab(label='Risk factors', children=[
            html.H3('Description of risk factors'),

            html.Div(
                className="four columns offset-by-one",
                children=[
                    html.Ul(id='risk-factors-list', children=[html.Li(i) for i in risk_factors])
                ]
            )
        ])
    ])
])

@app.callback(Output('priority-table','children'),
            [Input('submit-button','n_clicks'), Input('file-name', 'value')])
def update_datatable(n_clicks, file_name):
    if n_clicks:
        df = generate_prioritized_risk_list(data_path=file_name, return_df=True)
        data = df.to_dict('rows')
        columns = [{"name": i.replace('_', ' ').strip().capitalize(), "id": i,} for i in (df.columns)]

        return dt.DataTable(data=data, columns=columns, style_data={'whiteSpace': 'normal'},
            css=[{
                'selector': '.dash-cell div.dash-cell-value',
                'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
            }],
            style_as_list_view=True,
            style_header={
                'backgroundColor': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {
                        'column_id': 'quality_review',
                        'filter_query': '{quality_review} eq 1'
                    },
                    'backgroundColor': '#DC143C',
                    'color': 'white',
                }
            ],
            filter_action="native",
            sort_action="native",
            row_deletable=False,
            selected_rows=[],
            page_action="native",
            page_current= 0,
            page_size= 10,
        )


if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
