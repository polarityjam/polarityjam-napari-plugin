import napari
import numpy as np
from PyQt5.QtWidgets import QGraphicsPixmapItem, QComboBox, QGraphicsView, QGraphicsScene, QWidget, QVBoxLayout, \
    QPushButton, \
    QFileDialog, QLabel, QHBoxLayout, QSizePolicy


class JunctionAnnotationWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.scene = QGraphicsScene()

        # qt objects
        self.widgets = {
            "label_rp": QLabel("run PolarityJam on:"),
            "button": QPushButton("Run PolarityJam"),
            "label_jc": QLabel("Junction Class"),
            "dropdown": QComboBox(),
            "param_button": QPushButton("Parameter File")
        }

        # Set size policy of the widgets
        for widget in self.widgets.values():
            widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # Add items to the dropdown menu
        self.widgets["dropdown"].addItems(["Value 1", "Value 2", "Value 3", "Value 4", "Value 5"])

        # Create QHBoxLayout
        self.hbox_jc = QHBoxLayout()
        self.hbox_rp = QVBoxLayout()

        # Add label and button to the QVBoxLayout
        self.hbox_rp.addWidget(self.widgets["label_rp"])
        self.hbox_rp.addWidget(self.widgets["button"])
        self.hbox_rp.addWidget(self.widgets["param_button"])

        # Add label and dropdown to the QHBoxLayout
        self.hbox_jc.addWidget(self.widgets["label_jc"])
        self.hbox_jc.addWidget(self.widgets["dropdown"])

        self.widgets["param_button"].clicked.connect(self.load_parameter_file)

        self.widgets["button"].clicked.connect(self.run_polarityjam)

        self._build_layout()

    def _build_layout(self):
        self.layout.addLayout(self.hbox_rp)
        self.layout.addLayout(self.hbox_jc)

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
