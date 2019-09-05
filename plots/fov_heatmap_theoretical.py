from Objective import Objective

import os
import numpy as np
import plotly
import plotly.graph_objects as go

# Refractive index
n_e = 1

# Objective parameters
sample_depth = 900
working_distance = 5940
NA = 0.54

fov = np.linspace(-2500, 2500, 100)
xx, yy = np.meshgrid(fov, fov, sparse=False)
deviation = np.sqrt(xx ** 2 + yy ** 2)

objective = Objective(NA, working_distance, sample_depth, n_e)
efficiency = objective.theoretical_collection_efficiency(deviation, type='sphere')

fig = go.Figure(
    data=go.Heatmap(
        x=xx[0],
        y=yy[:, 0],
        z=efficiency,
        # zmin=0,
        # zmax=1
    ),
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
    ),
    title='Theoretical collection efficiency'
)

try:
    plotly.offline.plot(fig, filename='theoretical_heatmap.html')
except FileNotFoundError:
    os.mkdir('fov_heatmap_theoretical')
    plotly.offline.plot(fig, filename='theoretical_heatmap.html')
