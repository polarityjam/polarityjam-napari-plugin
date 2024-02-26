import napari
import numpy as np
from PyQt5.QtWidgets import QGraphicsPixmapItem, QComboBox, QGraphicsView, QGraphicsScene, QWidget, QVBoxLayout, \
    QPushButton, \
    QFileDialog, QLabel, QHBoxLayout, QSizePolicy, QLineEdit


class JunctionAnnotationWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.scene = QGraphicsScene()

        # qt objects
        self.widgets = {
            "label_input": QLabel("Input channels:"),
            "channel_junction_label": QLabel("channel_junction"),
            "channel_nucleus_label": QLabel("channel_nucleus"),
            "channel_organelle_label": QLabel("channel_organelle"),
            "channel_expression_marker_label": QLabel("channel_expression_marker"),
            "channel_junction": QLineEdit("-1"),
            "channel_nucleus": QLineEdit("-1"),
            "channel_organelle": QLineEdit("-1"),
            "channel_expression_marker": QLineEdit("-1"),
            "label_rp": QLabel("run PolarityJam on:"),
            "run_button": QPushButton("Run PolarityJam"),
            "label_jc": QLabel("Junction Class"),
            "dropdown": QComboBox(),
            "param_button": QPushButton("Parameter File")
        }

        # Set size policy of the widgets
        for widget in self.widgets.values():
            widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # Add items to the dropdown menu
        self.widgets["dropdown"].addItems(["Value 1", "Value 2", "Value 3", "Value 4", "Value 5"])

        # add connections
        self.widgets["param_button"].clicked.connect(self.load_parameter_file)
        self.widgets["run_button"].clicked.connect(self.run_polarityjam)

        # build layout
        self._build_layout()

    def _build_layout(self):
        # Create block-wise layout
        self.vbox_input = QVBoxLayout()
        self.vbox_run_pjam = QVBoxLayout()
        self.hbox_junction_labeling = QHBoxLayout()

        # Input block
        self.vbox_input.addWidget(self.widgets["label_input"])
        for channel in ["channel_junction", "channel_nucleus", "channel_organelle", "channel_expression_marker"]:
            hbox = QHBoxLayout()
            hbox.addWidget(self.widgets[channel + "_label"])
            hbox.addWidget(self.widgets[channel])
            self.vbox_input.addLayout(hbox)

        # Run PolarityJam block
        self.vbox_run_pjam.addWidget(self.widgets["label_rp"])
        self.vbox_run_pjam.addWidget(self.widgets["param_button"])
        self.vbox_run_pjam.addWidget(self.widgets["run_button"])

        # Junction labeling block
        self.hbox_junction_labeling.addWidget(self.widgets["label_jc"])
        self.hbox_junction_labeling.addWidget(self.widgets["dropdown"])

        # Add layouts to the overall layout
        self.layout.addLayout(self.vbox_input)
        self.layout.addLayout(self.vbox_run_pjam)
        self.layout.addLayout(self.hbox_junction_labeling)

    def run_polarityjam(self):
        # run polarityjam for the selected image
        pass

    def load_parameter_file(self):
        # Open a file dialog and load a YML file
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "YML Files (*.yml)")
        if file_path:
            print(f"Parameter file loaded: {file_path}")


def startup():
    viewer = napari.Viewer()
    image_layer = viewer.add_image(np.random.randint(0, 255, (64, 64)).astype(np.uint), name="My Image")

    widget = viewer.window.add_dock_widget(JunctionAnnotationWidget(), name="jatool")

    napari.run()


if __name__ == "__main__":
    startup()
