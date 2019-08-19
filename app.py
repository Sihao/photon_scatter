# -*- coding: utf-8 -*-
import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc

from Objective import Objective
from Medium import Medium

import numpy as np
import utils
import jsonpickle as pickle

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.layout = html.Div(children=[
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2("Simulation parameters"),
                            html.H6("Number of photons"),
                            dbc.Input(
                                placeholder="Number of photons",
                                type="number",
                                id="num_photons",
                                value=500,
                                step=100,
                                min=0,
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H6("Scattering coefficient (um^-1)"),
                                            dbc.Input(
                                                placeholder="Scattering coefficient",
                                                type="number",
                                                id="mu_s",
                                                value=0.01070,
                                                min=0,
                                                step=0.0001
                                            ),
                                        ]
                                    ),
                                    dbc.Col(
                                        [
                                            html.H6("Absorption coefficient (um^-1)"),
                                            dbc.Input(
                                                placeholder="Absorption coefficient",
                                                type="number",
                                                id="mu_a",
                                                value=0.002,
                                                min=0,
                                                step=0.0001
                                            ),
                                        ]
                                    )
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H6("Refractive index of sample"),
                                            dbc.Input(
                                                placeholder="Refractive index of sample",
                                                type="number",
                                                id="n_i",
                                                value=1.4,
                                                step=0.1,
                                                min=0
                                            ),
                                        ]
                                    ),
                                    dbc.Col(
                                        [
                                            html.H6("External refractive index"),
                                            dbc.Input(
                                                placeholder="External refractive index",
                                                type="number",
                                                id="n_e",
                                                value=1,
                                                step=0.1,
                                                min=0
                                            ),
                                        ]
                                    )
                                ]
                            ),
                            html.H6("Anisotropy"),
                            dbc.Input(
                                placeholder="Anisotropy",
                                type="number",
                                id="g",
                                value=0.94,
                                min=-1,
                                max=1,
                                step=0.01
                            ),
                            html.H6("Sample thickness (um)"),
                            dbc.Input(
                                placeholder="Sample thickness",
                                type="number",
                                id="sample_thickness",
                                value=900,
                                step=10,
                                min=0
                            ),
                            html.H6("Focus depth (distance from sample surface) (um)"),
                            dbc.Input(
                                placeholder="Focus depth",
                                type="number",
                                id="focus_depth",
                                value=890,
                                step=10,
                                min=0
                            ),

                            html.H6("Focal coordinates (um)"),
                            dbc.FormGroup(
                                [
                                    dbc.Label("X", width=1),
                                    dbc.Col(
                                        dbc.Input(
                                            type="number", id="x_cor", placeholder="X", value=0, step=10
                                        ),
                                        width=11,
                                    ),
                                ],
                                row=True,
                            ),
                            dbc.FormGroup(
                                [
                                    dbc.Label("Y",  width=1),
                                    dbc.Col(
                                        dbc.Input(
                                            type="number", id="y_cor", placeholder="Y", value=0, step=10
                                        ),
                                        width=11,
                                    ),
                                ],
                                row=True,
                            ),

                            html.H2("Objective properties"),
                            html.H6("Working distance (um)"),
                            dbc.Input(
                                placeholder="Working distance",
                                type="number",
                                id="working_distance",
                                value=5940,
                                min=0,
                                step=10
                            ),
                            html.H6("Numerical aperture"),
                            dbc.Input(
                                placeholder="Numerical aperture",
                                type="number",
                                id="NA",
                                value=0.54,
                                min=0,
                                step=0.01
                            ),
                            dbc.Button("Run simulation", id="run_sim", color="primary"),
                            html.Br(),
                            html.H2("Plot options"),
                            dbc.RadioItems(
                                options=[
                                    {'label': 'Points', 'value': 'points'},
                                    {'label': 'Cones', 'value': 'cones'},
                                ],
                                value='points',
                                id='points_cones'
                            ),
                            dbc.Checklist(
                                options=[
                                    {'label': 'Show objective aperture', 'value': True},
                                ],
                                value=[True],
                                id='show_aperture'
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dcc.Graph(
                                id='example-graph',
                                className="min-vh-100"

                            ),
                        ],
                        md=8
                    ),
                ],
                className="justify-content-center",
                no_gutters=True
            )
        ],
        style={'min-width': '100%', 'display': 'inline-block'},
        className=["pt-1", "pb-5", "px-5"]
    ),
    html.Div(id='photons', style={'display': 'none'})
], style={'margin': 0})


@app.callback(
    dash.dependencies.Output('photons', 'children'),
    [
        dash.dependencies.Input('run_sim', 'n_clicks'),
    ],
    [
        dash.dependencies.State('num_photons', 'value'),
        dash.dependencies.State('mu_s', 'value'),
        dash.dependencies.State('mu_a', 'value'),
        dash.dependencies.State('n_i', 'value'),
        dash.dependencies.State('n_e', 'value'),
        dash.dependencies.State('g', 'value'),
        dash.dependencies.State('sample_thickness', 'value'),
        dash.dependencies.State('focus_depth', 'value'),
        dash.dependencies.State('x_cor', 'value'),
        dash.dependencies.State('y_cor', 'value'),

    ],
)
def button_sim_photons(n_clicks, num_photons, mu_s, mu_a, n_i, n_e, g, sample_thickness, focus_depth, x_cor, y_cor):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    start_pos = np.array([x_cor, y_cor, focus_depth])
    medium_shape = np.array([float('inf'), float('inf'), sample_thickness])
    medium = Medium(medium_shape, mu_s, mu_a, n_i, n_e, g)

    photons = utils.multiple_sim(medium, start_pos, num_photons, omit_bottom=False)

    return pickle.dumps(photons)


@app.callback(
    dash.dependencies.Output('example-graph', 'figure'),
    [
        dash.dependencies.Input('photons', 'children'),
        dash.dependencies.Input('points_cones', 'value'),
        dash.dependencies.Input('show_aperture', 'value')
    ],
    [
        dash.dependencies.State('sample_thickness', 'value'),
        dash.dependencies.State('working_distance', 'value'),
        dash.dependencies.State('NA', 'value'),
    ]
)
def button_plot_photons(photons_json, points_cones, show_aperture, sample_thickness, working_distance, NA):
    objective = Objective(NA, working_distance, sample_thickness)
    photons = pickle.loads(photons_json)
    if points_cones == 'points':
        fig = utils.plot_photons(photons, objective, show_aperture=show_aperture, cones=False)
    else:
        fig = utils.plot_photons(photons, objective, show_aperture=show_aperture, cones=True)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
