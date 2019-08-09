import numpy as np
from itertools import compress
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Photon import Photon


def fov_sim(medium, fov, num_photons, depth, omit_bottom=False):
    """
    Simulate photon propagation over a grid of start positions. Only supports square grids.
    :param medium: Medium object
    :param fov: List of 1-D coordinates which will be made into square 2-D grid.
    :param num_photons: Number of photons to simulate per position in the grid
    :param depth: Flag to omit all photons exiting from the bottom of the medium. Default is `False`.
    :param omit_bottom: Flag to only propagate photon for one step. Default is `False`.
    :return: Nested list of Photon objects. Main list contains lists of photons with start positions on the same row.
             Sublist contains lists of Photon objects per position in row.
    """
    result = []
    for i in fov:
        result_j = []
        for j in fov:
            fov_row = [single_sim(medium, np.array([i, j, depth]), omit_bottom) for _ in range(num_photons)]
            fov_row = list(filter(None, fov_row))
            result_j.append(fov_row)
        result.append(result_j)

    return result


def multiple_sim(medium, start_pos, num_photons, omit_bottom=False, single_step=False):
    """
    Simulate multiple photons propagating through a medium
    :param medium: Medium object
    :param start_pos: Position the photon is starting from. Must be within the Medium as defined in Medium.shape
    :param num_photons: Number of photons to simulate
    :param omit_bottom: Flag to omit all photons exiting from the bottom of the medium. Default is `False`.
    :param single_step: Flag to only propagate photon for one step. Default is `False`.
    :return: List of Photon objects. Omitted and absorbed photons are filtered out.
    """
    photons = [single_sim(medium, start_pos, omit_bottom, single_step) for _ in range(num_photons)]
    photons = list(filter(None, photons))

    return photons


def single_sim(medium, start_pos, omit_bottom=False, single_step=False):
    """
    Simulate a single photon propagating through a medium
    :param medium: Medium object
    :param start_pos: Position the photon is starting from. Must be within the Medium as defined in Medium.shape
    :param omit_bottom: Flag to omit all photons exiting from the bottom of the medium. Default is `False`.
    :param single_step: Flag to only propagate photon for one step. Default is `False`.
    :return: Photon object or `None` if photon is omitted or absorbed
    """
    photon = Photon(start_pos, medium)

    if single_step is True:
        photon.propagate(omit_bottom)
    else:
        while photon.is_propagating and not photon.is_absorbed:
            photon.propagate(omit_bottom)
    # print('Path length: %f' %photon.total_path)

    if photon.is_omitted or photon.is_absorbed:
        return None
    else:
        return photon


def calc_acceptance_ratio(photons, objective):
    """
    Calculate ratio of photons accepted into the objective
    :param photons: List of Photon objects.
    :param objective: Objective object
    :return: Scalar value
    """
    # Filter accepted photons
    acceptance_list = list(map(objective.photon_accepted, photons))
    accepted_photons = list(compress(photons, acceptance_list))
    accepted_positions = np.array([photon.path[-1] for photon in accepted_photons])

    ratio = len(accepted_positions) / len(photons)

    return ratio


def calc_acceptance_matrix(fov_photon_matrix, objective):
    """
    Calculate ratio of photons accepted into the objective on a grid
    :param fov_photon_matrix: Output of fov_sim()
    :param objective: Objective object
    :return: Numpy array where each element holds acceptance ratio at a given position
    """
    acceptance_matrix = np.reshape(
        np.array(
            [calc_acceptance_ratio(fov_spot, objective) for fov_row in fov_photon_matrix for fov_spot in
             fov_row]),
        (len(fov_photon_matrix), len(fov_photon_matrix[0])))

    return acceptance_matrix


def plot_photons(photons, objective, show_aperture=False, cones=False):
    """
    Plot positions of photons in 3-D. Colours accepted photons in green and rejected photons in red.
    :param photons: List of Photon objects
    :param objective: Objective object
    :param show_aperture: Flag to plot circle representing front aperture of objective. Default is `False`.
    :param cones: Flag to plot photons as dots or cones (indicating propagation direction). Default is `False`.
    :return: Plotly Figure object.
    """
    # Filter accepted photons
    acceptance_list = list(map(objective.photon_accepted, photons))
    accepted_photons = list(compress(photons, acceptance_list))

    # Get coordinates of accepted photons
    accepted_positions = np.array([photon.path[-1] for photon in accepted_photons])

    # Filter rejected photons
    rejected_photons = list(compress(photons, [not photon for photon in acceptance_list]))

    # Get coordinates of rejected photons
    rejected_positions = np.array([photon.path[-1] for photon in rejected_photons])

    if cones is False:
        try:
            fig = go.Figure(
                data=[go.Scatter3d(x=accepted_positions[:, 0], y=accepted_positions[:, 1], z=accepted_positions[:, 2],
                                   mode='markers', marker=dict(
                        size=3,
                        opacity=0.8,
                        color='green'
                    ))])
            fig.add_trace(
                go.Scatter3d(x=rejected_positions[:, 0], y=rejected_positions[:, 1], z=rejected_positions[:, 2],
                             mode='markers', marker=dict(
                        size=3,
                        opacity=0.2,
                        color='red'
                    )))
        except IndexError:
            fig = go.Figure(
                data=[go.Scatter3d(x=rejected_positions[:, 0], y=rejected_positions[:, 1], z=rejected_positions[:, 2],
                                   mode='markers', marker=dict(
                        size=3,
                        opacity=0.2,
                        color='red'
                    ))]
            )

    else:
        # Define constant colorscale
        pl_red = [[0, '#bd1540'],
                  [1, '#bd1540']]

        pl_green = [[0, '#009900'],
                   [1, '#009900']]
        try:
            fig = go.Figure(
                data=[go.Cone(x=accepted_positions[:, 0], y=accepted_positions[:, 1], z=accepted_positions[:, 2],
                                  u=[photon.mu_x for photon in accepted_photons],
                                  v=[photon.mu_y for photon in accepted_photons],
                                  w=[photon.mu_z for photon in accepted_photons],
                                  anchor="tail",
                                  colorscale=pl_green,
                                  hoverinfo="all",
                                  showscale=False,
                                  sizeref=3)]
            )
            fig.add_trace(go.Cone(x=rejected_positions[:, 0], y=rejected_positions[:, 1], z=rejected_positions[:, 2],
                                  u=[photon.mu_x for photon in rejected_photons],
                                  v=[photon.mu_y for photon in rejected_photons],
                                  w=[photon.mu_z for photon in rejected_photons],
                                  anchor="tail",
                                  colorscale=pl_red,
                                  hoverinfo="all",
                                  showscale=False,
                                  sizeref=3)
            )

        except IndexError:
            fig = go.Figure(
                data=[go.Scatter3d(x=rejected_positions[:, 0], y=rejected_positions[:, 1], z=rejected_positions[:, 2],
                                   mode='markers', marker=dict(
                        size=3,
                        opacity=0.2,
                        color='red'
                    ))]
            )

    if show_aperture:
        # Show objective aperture
        aperture_z = (np.cos(objective.theta) * objective.working_distance + objective.sample_thickness) * np.ones(
            (50, 50))
        # Need multiple radii to plot surface
        R = np.linspace(0, objective.front_aperture / 2, 50)

        # Sample angles of circle
        u = np.linspace(0, 2 * np.pi, 50)

        # Calculate x-y coordinates for circle
        x = np.outer(R, np.cos(u))
        y = np.outer(R, np.sin(u))

        # Add to plot
        fig.add_trace(go.Surface(
            x=x,
            y=y,
            z=aperture_z,
            showscale=False)
        )

    fig.update_layout(
        title=go.layout.Title(
            text="Photon positions",
            xref="paper",
            x=0
        ),
        scene=dict(
            xaxis_title="x (um)",
            yaxis_title="y (um)",
            zaxis_title="z (um)",
        )
    )

    return fig


def plot_fov_heatmap(acceptance_matrix, fov):
    """
    Plot heatmap of acceptance ratio across a field of view.
    :param acceptance_matrix: Numpy array where each element holds acceptance ratio at a given position
    :param fov: List of 1-D coordinates. Used for axis range of heatmap.
    :return: Plotly Figure object.
    """
    fig = go.Figure(
        data=go.Heatmap(
            x=fov,
            y=fov,
            z=acceptance_matrix,
        )
    )

    fig.update_layout(
        autosize=False,
        width=800,
        height=800,
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            title="y (um)"
        ),
        xaxis=dict(
            title="x (um)"
        )
    )

    return fig


def plot_photon_path(photon):
    """
    Plot propagation path of a single photon in 3-D.
    :param photon: Photon object
    :return: Plotly Figure object.
    """
    # fig = go.Figure(
    #     data=[go.Scatter3d(
    #         x=photon.path[:, 0],
    #         y=photon.path[:, 1],
    #         z=photon.path[:, 2],
    #         mode='lines+markers',
    #         marker=dict(
    #             size=3,
    #             opacity=0.8,
    #             color=np.array(range(0, len(photon.path))) / (len(photon.path) - 1),
    #             # set color to an array/list of desired values
    #             colorscale='Viridis',  # choose a colorscale
    #             colorbar=dict(
    #                 thickness=20,
    #                 tickmode='array',
    #                 tickvals=[0, 1],
    #                 ticktext=["Start", "End"])
    #         ),
    #         line=dict(
    #             color='#1f77b4',
    #             width=5)
    #     )]
    # )

    fig = go.Figure(
        data=[go.Cone(x=photon.path[:, 0],
                      y=photon.path[:, 1],
                      z=photon.path[:, 2],
                      u=[direction_cosine[0] for direction_cosine in photon.direction_cosines],
                      v=[direction_cosine[1] for direction_cosine in photon.direction_cosines],
                      w=[direction_cosine[2] for direction_cosine in photon.direction_cosines],
                      anchor="tail",
                      colorscale="Viridis",
                      hoverinfo="all",
                      showscale=False,
                      sizemode="absolute",
                      sizeref=3)]
    )

    fig.add_trace(go.Scatter3d(
            x=photon.path[:, 0],
            y=photon.path[:, 1],
            z=photon.path[:, 2],
            mode='lines',
            marker=dict(
                size=2,
                opacity=0.8,
                color=np.array(range(0, len(photon.path))) / (len(photon.path) - 1),
                # set color to an array/list of desired values
                colorscale='Viridis',  # choose a colorscale
                colorbar=dict(
                    thickness=20,
                    tickmode='array',
                    tickvals=[0, 1],
                    ticktext=["Start", "End"])
            ),
            line=dict(
                color='#1f77b4',
                width=5)
        )
    )

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter(photon.path[:, 0], photon.path[:, 1], photon.path[:, 2], s=5)
    # ax.plot(photon.path[:, 0], photon.path[:, 1], photon.path[:, 2])

    return fig


def plot_axial_paths(photons, medium, objective):
    """
    Plot the path of photons in 2-D (both in the X-Z and Y-Z plane)
    :param photons: List of Photon objects.
    :param medium: Medium object.
    :param objective: Objective object.
    :return: PLotly Figure object.
    """
    # Filter accepted photons
    acceptance_list = list(map(objective.photon_accepted, photons))
    accepted_photons = list(compress(photons, acceptance_list))

    # Filter rejected photons
    rejected_photons = list(compress(photons, [not photon for photon in acceptance_list]))

    fig = make_subplots(rows=1, cols=2, subplot_titles=["X-Z Plane", "Y-Z Plane"], shared_yaxes=True)
    fig.update_layout(
        title=go.layout.Title(
            text="Photon paths",
            xref="paper",
            x=0
        ),


        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text="Depth (um)",
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="#7f7f7f"
                )
            )
        )
    )
    fig.update_xaxes(
        title=go.layout.xaxis.Title(
            text="x Axis (um)",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        ),
        row=1,
        col=1

    )
    fig.update_xaxes(
        title=go.layout.xaxis.Title(
            text="y Axis (um)",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        ),
        row=1,
        col=2

    )

    for i, photon in enumerate(accepted_photons):
        fig.add_trace(go.Scatter(x=photon.path[:, 0], y=photon.path[:, 2],
                                 mode='lines',
                                 name=('Accepted photon %i' % i),
                                 line=dict(
                                     color='SeaGreen'
                                 )
                                 ),
                      row=1, col=1,
                      )
        fig.add_trace(go.Scatter(x=photon.path[:, 1], y=photon.path[:, 2],
                                 mode='lines',
                                 name=('Accepted photon %i' % i),
                                 line=dict(
                                     color='SeaGreen'
                                 )
                                 ),
                      row=1, col=2,
                      )

    for i, photon in enumerate(rejected_photons):
        fig.add_trace(go.Scatter(x=photon.path[:, 0], y=photon.path[:, 2],
                                 mode='lines',
                                 name=('Rejected photon %i' % i),
                                 line=dict(
                                     color='Crimson'
                                 )
                                 ),
                      row=1, col=1,
                      )
        fig.add_trace(go.Scatter(x=photon.path[:, 1], y=photon.path[:, 2],
                                 mode='lines',
                                 name=('Rejected photon %i' % i),
                                 line=dict(
                                     color='Crimson'
                                 )
                                 ),
                      row=1, col=2,
                      )

    # Get minimum and maximum x values from photon paths
    x0_0 = min([min(photon.path[:, 0]) for photon in photons])
    x1_0 = max([max(photon.path[:, 0]) for photon in photons])

    # Get minimum and maximum y values from photon paths
    x0_1 = min([min(photon.path[:, 1]) for photon in photons])
    x1_1 = max([max(photon.path[:, 1]) for photon in photons])

    fig.update_layout(
        shapes=[
            # Fill sample area in XZ plot
            go.layout.Shape(
                type="rect",
                x0=x0_0 + x0_0 * 0.3,
                y0=0,
                x1=x1_0 + x1_0 * 0.3,
                y1=medium.shape[2],
                line=dict(
                    width=0,
                ),
                fillcolor="LightSkyBlue",
                opacity=.2,
            ),
            # Fill sample area in YZ plot
            go.layout.Shape(
                type="rect",
                x0=x0_1 + x0_1 * 0.3,
                y0=0,
                x1=x1_1 + x1_1 * 0.3,
                xref="x2",
                y1=medium.shape[2],
                yref='y2',
                line=dict(
                    width=0,
                ),
                fillcolor="LightSkyBlue",
                opacity=0.2
            ),
        ],
    )

    # Annotate shapes
    fig.add_trace(
        go.Scatter(
            x=[x0_0 + x0_0 * 0.15],
            y=[medium.shape[2] - medium.shape[2] * 0.15],
            text=["Sample"],
            mode="text",
            textposition="bottom right",
            showlegend=False,
            ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=[x0_1 + x0_1 * 0.15],
            y=[medium.shape[2] - medium.shape[2] * 0.15],
            text=["Sample"],
            mode="text",
            textposition="bottom right",
            showlegend=False,
            ),
        row=1,
        col=2
    )

    return fig


def animate_photon_positions(photons, objective):
    def get_photon_coordinates(input_photons, axis, step):
        coordinates = []

        for i, photon in enumerate(input_photons):
            try:
                coordinates.append(photon.path[step][axis])
            except IndexError:
                coordinates.append(photon.path[-1][axis])

        return coordinates

    max_steps = max([len(photon.path) for photon in photons])

    # Filter accepted photons
    acceptance_list = list(map(objective.photon_accepted, photons))
    accepted_photons = list(compress(photons, acceptance_list))

    # Filter rejected photons
    rejected_photons = list(compress(photons, [not photon for photon in acceptance_list]))

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=get_photon_coordinates(accepted_photons, 0, 0),
                y=get_photon_coordinates(accepted_photons, 1, 0),
                z=get_photon_coordinates(accepted_photons, 2, 0),
                mode='markers',
                marker=dict(
                    size=3,
                    color="green"
                )
            ),
            go.Scatter3d(
                x=get_photon_coordinates(rejected_photons, 0, 0),
                y=get_photon_coordinates(rejected_photons, 1, 0),
                z=get_photon_coordinates(rejected_photons, 2, 0),
                mode='markers',
                marker=dict(
                    size=3,
                    color="red"
                )
            ),

        ],
        layout=go.Layout(
            scene=dict(

                xaxis=dict(range=[-3000, 3000], autorange=False),
                yaxis=dict(range=[-3000, 3000], autorange=False),
                zaxis=dict(range=[-500, 2500], autorange=False),
                xaxis_title="x (um)",
                yaxis_title="y (um)",
                zaxis_title="z (um)",
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            title="Photon propagation animation",
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(
                                    duration=50,
                                    redraw=True
                                ),
                                fromcurrent=True,
                                mode="immediate"

                            )
                        ]
                    ),
                    dict(
                     args=[
                         [None],
                         dict(
                             frame=dict(
                                 duration=0,
                                 redraw=False
                             ),
                             mode="immediate",
                         )

                     ],
                     label="Stop",
                     method="animate"
                    )
                ])]
        ),
        frames=[

                go.Frame(
                    data=[
                        go.Scatter3d(
                            x=get_photon_coordinates(accepted_photons, 0, i),
                            y=get_photon_coordinates(accepted_photons, 1, i),
                            z=get_photon_coordinates(accepted_photons, 2, i),
                            marker=dict(
                                color="green"
                            ),
                            name="Accepted"
                        ),
                        go.Scatter3d(
                            x=get_photon_coordinates(rejected_photons, 0, i),
                            y=get_photon_coordinates(rejected_photons, 1, i),
                            z=get_photon_coordinates(rejected_photons, 2, i),
                            marker=dict(
                                color="red"
                            ),
                            name="Rejected"
                        )
                    ]
                )
                for i in range(max_steps)
        ]
    )

    return fig
