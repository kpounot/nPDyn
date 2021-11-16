"""Helper class to generate an interactive legend in matplotlib."""


import numpy as np
import matplotlib.pyplot as plt


class InteractiveLegend:
    def __init__(self, fig):
        self.fig = fig

        self.axes = (
            fig.axes
            if isinstance(fig.axes, (list, np.ndarray))
            else [fig.axes]
        )

        self.legends = [ax.get_legend() for ax in self.axes]
        for leg in self.legends:
            if leg is not None:
                leg.set_draggable(True)

        self.lookup_artist, self.lookup_handle = self._build_lookups()
        self._setup_connections()

        self.update()

    def _setup_connections(self):
        for leg in self.legends:
            if leg is not None:
                for artist in leg.texts + leg.legendHandles:
                    artist.set_picker(10)  # 10 points tolerance

        self.fig.canvas.mpl_connect("pick_event", self.on_pick)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def _build_lookups(self):
        lookup_artist = {}
        lookup_handle = {}

        for idx, ax in enumerate(self.axes):
            legend = self.legends[idx]
            if legend is not None:
                labels = [t.get_text() for t in legend.texts]
                handles = legend.legendHandles
                label2handle = dict(zip(labels, handles))
                handle2text = dict(zip(handles, legend.texts))
                for artist in ax.axes.get_legend_handles_labels()[0]:
                    if artist.get_label() in labels:
                        handle = label2handle[artist.get_label()]
                        lookup_handle[artist] = handle
                        lookup_artist[handle] = artist
                        lookup_artist[handle2text[handle]] = artist

                lookup_handle.update(zip(handles, handles))
                lookup_handle.update(zip(legend.texts, handles))

        return lookup_artist, lookup_handle

    def on_pick(self, event):
        handle = event.artist
        if handle in self.lookup_artist:
            artist = self.lookup_artist[handle]
            if len(artist.get_children()) == 0:
                artist.set_visible(not artist.get_visible())
            else:  # case of Container object with multiple artists
                for artist in artist.get_children():
                    artist.set_visible(not artist.get_visible())
            self.update()

    def on_click(self, event):
        if event.button == 3:
            visible = False
        elif event.button == 2:
            visible = True
        else:
            return

        for artist in self.lookup_artist.values():
            if len(artist.get_children()) == 0:
                artist.set_visible(visible)
            else:  # case of Container object with multiple artists
                for artist in artist.get_children():
                    artist.set_visible(visible)
        self.update()

    def update(self):
        for artist in self.lookup_artist.values():
            handle = self.lookup_handle[artist]
            visible = True
            if len(artist.get_children()) == 0:
                visible = artist.get_visible()
            else:
                visible = artist.get_children()[0].get_visible()

            handle.set_visible(visible)
        self.fig.canvas.draw()

    def show(self):
        plt.show()
