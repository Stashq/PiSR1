from typing import Dict, List

import plotly.graph_objects as go
from IPython.display import Image


class Plot:

    def __init__(self):
        super(Plot, self).__init__()

    def convergence(
        self,
        losses: List[List[float]],
        names: List[str],
        xaxis_title: str = '',
        yaxis_title: str = ''
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
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title
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

    def histogram(
        self,
        x: List[float],
        title: str,
        xaxis_title: str = '',
        yaxis_title: str = '',
        use_log_scale: bool = False
    ):
        fig = go.Figure()

        histogram = go.Histogram(x=x)

        fig.add_trace(histogram)

        if use_log_scale:
            fig.update_yaxes(type="log")

        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title
        )

        fig.show(renderer='svg')

    def bar(
        self,
        data: Dict[str, float],
        title: str,
        xaxis_title: str = '',
        yaxis_title: str = '',
    ):
        fig = go.Figure()

        x, y = data.keys(), data.values()
        xy = zip(x, y)
        xy = sorted(xy, key=lambda key_value: key_value[0])
        x, y = zip(*xy)

        bar = go.Bar(
            x=x,
            y=y,
            text=[f'{value:.2f}' for value in y],
            textposition='auto',
        )

        fig.add_trace(bar)

        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title
        )

        fig.show(renderer='svg')
