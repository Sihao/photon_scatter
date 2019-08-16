from Objective import Objective
from Medium import Medium

import os
import numpy as np
import plotly
import utils
import plotly.graph_objects as go

# Scattering coefficient
# in um^-1
mu_s = 0.01070

# Absorption coefficient
mu_a = 0.0002

# Refractive indices
n_i = 1.4

# in air
n_e = 1

# Anisotropy factor
g = 0.94

sample_depth = 900
working_distance = 5940
NA = 0.54

medium_shape = np.array([float('inf'), float('inf'), sample_depth])
medium = Medium(medium_shape, mu_s, mu_a, n_i, n_e, g)

num_photons = 7000

start_positions = [
    np.array([2500, 2500, 890]),
    np.array([2500, 2500, 880]),
    np.array([2500, 2500, 870]),
    np.array([2500, 2500, 860]),
    np.array([2500, 2500, 850]),
    np.array([2500, 2500, 840]),
    np.array([2500, 2500, 830]),
    np.array([2500, 2500, 820]),
    np.array([2500, 2500, 810]),
    np.array([2500, 2500, 800]),
    np.array([2500, 2500, 790]),
    np.array([2500, 2500, 780]),
    np.array([2500, 2500, 770]),
    np.array([2500, 2500, 760]),
    np.array([2500, 2500, 750]),
    np.array([2500, 2500, 740]),
    np.array([2500, 2500, 730]),
    np.array([2500, 2500, 720]),
    np.array([2500, 2500, 710]),
    np.array([2500, 2500, 700]),
    np.array([2500, 2500, 600]),
    np.array([2500, 2500, 500]),
    np.array([2500, 2500, 400]),
    np.array([2500, 2500, 300]),
    np.array([2500, 2500, 200]),
    np.array([2500, 2500, 100]),
    np.array([2500, 2500, 90]),
    np.array([2500, 2500, 80]),
    np.array([2500, 2500, 70]),
    np.array([2500, 2500, 60]),
    np.array([2500, 2500, 50]),
    np.array([2500, 2500, 40]),
    np.array([2500, 2500, 30]),
    np.array([2500, 2500, 20]),
    np.array([2500, 2500, 10]),
    np.array([2500, 2500, 0]),

    np.array([2500, 2500, 890]),
    np.array([2500, 2500, 850]),
    np.array([2500, 2500, 700]),
    np.array([2500, 2500, 250]),
    np.array([2500, 2500, 0])
]

ratios = []
for start_pos in start_positions:
    distance_to_sample = working_distance + start_pos[2]

    objective = Objective(NA, working_distance, sample_depth)

    photons = utils.multiple_sim(medium, start_pos, num_photons, omit_bottom=False)

    # Plot photon positions
    fig = utils.plot_photons(photons, objective)

    try:
        plotly.offline.plot(fig, filename='depth_plots/(%s,%s,%s).html' % (start_pos[0], start_pos[1], start_pos[2]))
    except FileNotFoundError:
        os.mkdir('depth_plots')
        plotly.offline.plot(fig, filename='depth_plots/(%s,%s,%s).html' % (start_pos[0], start_pos[1], start_pos[2]))

    # Plot axial paths
    fig = utils.plot_axial_paths(photons, medium, objective)

    try:
        plotly.offline.plot(
            fig, filename='depth_plots/axial_path(%s,%s,%s).html' % (start_pos[0], start_pos[1], start_pos[2])
        )
    except FileNotFoundError:
        os.mkdir('depth_plots')
        plotly.offline.plot(
            fig, filename='depth_plots/axial_path(%s,%s,%s).html' % (start_pos[0], start_pos[1], start_pos[2])
        )

    ratios.append(utils.calc_acceptance_ratio(photons, objective))

fig = go.Figure(
    data=go.Scatter(
        mode='lines',
        x=[900 - pos[2] for pos in start_positions],
        y=ratios
    ),
    layout=go.Layout(
        title='Acceptance ratio with respect to depth of excitation',
        xaxis=dict(
            title='Depth from surface (um)',
        ),
        yaxis=dict(
            title='Acceptance ratio (%%)'
        )
    )
)

try:
    plotly.offline.plot(
        fig, filename='depth_plots/acceptance_v_depth_off-axis.html'
    )

except FileNotFoundError:
    os.mkdir('depth_plots')
    plotly.offline.plot(
        fig, filename='depth_plots/acceptance_v_depth_off-axis.html'
    )
