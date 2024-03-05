import napari
import numpy as np

from jat.ui.napari_jat_plugin import JunctionAnnotationWidget


def startup():
    viewer = napari.Viewer()

    widget = viewer.window.add_dock_widget(JunctionAnnotationWidget(viewer), name="jatool")

    napari.run()


if __name__ == "__main__":
    startup()
