import napari
import numpy as np

from jat.ui.napari_jat_plugin import JunctionAnnotationWidget


def startup():
    viewer = napari.Viewer()
    image_layer = viewer.add_image(np.random.randint(0, 255, (64, 64)).astype(np.uint), name="My Image")

    widget = viewer.window.add_dock_widget(JunctionAnnotationWidget(), name="jatool")

    napari.run()


if __name__ == "__main__":
    startup()
