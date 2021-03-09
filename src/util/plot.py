from typing import List

import plotly.graph_objects as go


class Plot:

    def __init__(self):
        super(Plot, self).__init__()

    def convergence(
        self,
        losses: List[List[float]],
        names: List[str]
    ):
        fig = go.Figure()

        for loss, name in zip(losses, names):
            trace = go.Scatter(
                x=list(range(0, len(loss))),
                y=loss,
                mode='lines',
                name=name
            )

            fig.add_trace(trace)

        fig.update_layout(title='Convergence')
        fig.update_xaxes(title_text='Epochs')
        fig.update_yaxes(title_text='MSE Loss')

        fig.show()
        # return fig
