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
            trace = go.Scatter(
                x=list(range(0, len(loss))),
                y=loss,
                mode='lines',
                name=name
            )

            fig.add_trace(trace)

        fig.update_layout(
            title='Convergence',
            xaxis_title='Epochs',
            yaxis_title='MSE Loss'
        )

        fig.show(renderer='svg')
        # img_bytes = fig.to_image(format='png', scale=5)
        # return Image(img_bytes)
