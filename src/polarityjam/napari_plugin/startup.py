import napari

from polarityjam.napari_plugin.ui.napari_pjam_plugin import PjamNapariWidget


def run_napari(_):
    viewer = napari.Viewer()

    widget = viewer.window.add_dock_widget(PjamNapariWidget(viewer), name="Polarity-JaM Napari")

    napari.run()


if __name__ == "__main__":
    run_napari("")
