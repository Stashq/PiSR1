from typing import List

import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as pyo


class Plot:

    def __init__(self, image_width: int = 1200, image_height: int = 900):
        super(Plot, self).__init__()

        self.IMAGE_WIDTH = image_width
        self.IMAGE_HEIGHT = image_height

        pyo.init_notebook_mode(connected=False)
        pio.renderers.default = 'notebook'

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

        fig.update_layout(
            title='Convergence',
            xaxis_title='Epochs',
            yaxis_title='MSE Loss',
        )

        pyo.iplot(
            fig,
            # filename='filename',
            image_width=self.IMAGE_WIDTH,
            image_height=self.IMAGE_HEIGHT
        )
        # fig.show()
