import numpy as np
from itertools import compress
import plotly.graph_objects as go

from Photon import Photon


def parallel_sim(medium, num_photons):
    result = [single_sim(medium) for _ in range(num_photons)]

    return result


def fov_sim(medium, fov, num_photons, depth, omit_bottom=False):
    result = []
    for i in fov:
        result_j = []
        for j in fov:
            fov_row = [single_sim(medium, np.array([i, j, depth]), omit_bottom) for _ in range(num_photons)]
            fov_row = list(filter(None, fov_row))
            result_j.append(fov_row)
        result.append(result_j)

    return result


def single_sim(medium, start_pos, omit_bottom=False):
    photon = Photon(start_pos, medium)

    while photon.is_propagating and not photon.is_absorbed:
        photon.propagate(omit_bottom)
    # print('Path length: %f' %photon.total_path)

    if photon.is_omitted or photon.is_absorbed:
        return None
    else:
        return photon


def calc_acceptance_ratio(photons, objective):
    # Filter accepted photons
    acceptance_list = list(map(objective.photon_accepted, photons))
    accepted_photons = list(compress(photons, acceptance_list))
    accepted_positions = np.array([photon.path[-1] for photon in accepted_photons])

    ratio = len(accepted_positions) / len(photons)

    return ratio


def calc_acceptance_matrix(fov_photon_matrix, objective):
    acceptance_matrix = np.reshape(
        np.array(
            [calc_acceptance_ratio(fov_spot, objective) for fov_row in fov_photon_matrix for fov_spot in
             fov_row]),
        (len(fov_photon_matrix), len(fov_photon_matrix[0])))

    return acceptance_matrix


def plot_photons(photons, objective, show_aperture=False):
    # Filter accepted photons
    acceptance_list = list(map(objective.photon_accepted, photons))
    accepted_photons = list(compress(photons, acceptance_list))

    # Get coordinates of accepted photons
    accepted_positions = np.array([photon.path[-1] for photon in accepted_photons])

    # Filter rejected photons
    rejected_photons = list(compress(photons, [not photon for photon in acceptance_list]))

    # Get coordinates of rejected photons
    rejected_positions = np.array([photon.path[-1] for photon in rejected_photons])

    try:
        fig = go.Figure(
            data=[go.Scatter3d(x=accepted_positions[:, 0], y=accepted_positions[:, 1], z=accepted_positions[:, 2],
                               mode='markers', marker=dict(
                    size=3,
                    opacity=0.8,
                    color='green'
                ))])
        fig.add_trace(go.Scatter3d(x=rejected_positions[:, 0], y=rejected_positions[:, 1], z=rejected_positions[:, 2],
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

    return fig


def plot_fov_heatmap(acceptance_matrix, fov):
    fig = go.Figure(data=go.Heatmap(
        x=fov,
        y=fov,
        z=acceptance_matrix,)
    )

    fig.update_layout(yaxis=dict(
        scaleanchor="x",
        scaleratio=1,)
    )

    return fig


def plot_photon_path(photon):
    fig = go.Figure(
        data=[go.Scatter3d(
            x=photon.path[:, 0],
            y=photon.path[:, 1],
            z=photon.path[:, 2],
            mode='lines+markers',
            marker=dict(
                size=3,
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
        )]
    )

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter(photon.path[:, 0], photon.path[:, 1], photon.path[:, 2], s=5)
    # ax.plot(photon.path[:, 0], photon.path[:, 1], photon.path[:, 2])

    return fig
