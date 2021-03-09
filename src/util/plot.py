from typing import List

import plotly.graph_objects as go
from IPython.display import Image


class Plot:

    def __init__(self):
        super(Plot, self).__init__()

    def convergence(
        self,
        losses: List[List[float]],
        names: List[str]
    ) -> Image:
        fig = go.Figure()

        for loss, name in zip(losses, names):
            scatter = go.Scatter(
                x=list(range(0, len(loss))),
                y=loss,
                mode='lines',
                name=name
            )

            fig.add_trace(scatter)

        fig.update_layout(
            title='Convergence',
            xaxis_title='Epochs',
            yaxis_title='MSE Loss'
        )

        fig.show(renderer='svg')

    def relplot(
        self,
        x: List[float],
        y: List[float],
        marker_color: List[float],
        title: str,
        names: List[str]
    ):
        fig = go.Figure()

        scatter = go.Scatter(
            x=x,
            y=y,
            marker_color=marker_color,
            mode='markers',
            marker=dict(
                opacity=0.5,
                showscale=True,
                colorbar=dict(
                    title=names[2]
                )
            )
        )

        fig.add_trace(scatter)

        fig.update_layout(
            title=title,
            xaxis_title=names[0],
            yaxis_title=names[1]
        )

        fig.show(renderer='svg')

    def histogram2d(
        self,
        x: List[float],
        y: List[float],
        title: str,
        names: List[str]
    ):
        fig = go.Figure()

        histogram = go.Histogram2d(
            x=x,
            y=y,
            colorbar=dict(
                title=names[2]
            )
        )
        fig.add_trace(histogram)

        fig.update_layout(
            title=title,
            xaxis_title=names[0],
            yaxis_title=names[1]
        )

        fig.show(renderer='svg')
