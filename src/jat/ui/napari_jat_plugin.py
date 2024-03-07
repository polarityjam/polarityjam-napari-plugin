import napari
import numpy as np
import pkg_resources

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGraphicsScene, QLabel, QLineEdit, QPushButton, QComboBox, \
    QSizePolicy, QHBoxLayout, QFileDialog, QMessageBox, QSpinBox

from polarityjam import Extractor, PropertiesCollection, load_segmenter
from polarityjam import RuntimeParameter, PlotParameter, SegmentationParameter, ImageParameter

import os


class JunctionAnnotationWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.params_image = ImageParameter()
        # reset
        self.params_image.channel_junction = -1
        self.params_image.channel_nucleus = -1
        self.params_image.channel_organelle = -1
        self.params_image.channel_expression_marker = -1
        self.params_image.pixel_to_micron_ratio = 1.0

        self.params_runtime = RuntimeParameter()
        self.params_plot = PlotParameter()
        self.params_seg = None
        self.output_path = os.getcwd()
        self.output_path_prefix = "output"

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.scene = QGraphicsScene()

        # qt objects
        self.widgets = {
            # Input block
            "label_input": QLabel("Input channels:"),
            "channel_junction_label": QLabel("channel_junction"),
            "channel_nucleus_label": QLabel("channel_nucleus"),
            "channel_organelle_label": QLabel("channel_organelle"),
            "channel_expression_marker_label": QLabel("channel_expression_marker"),
            "channel_junction": QSpinBox(),
            "channel_nucleus": QSpinBox(),
            "channel_organelle": QSpinBox(),
            "channel_expression_marker": QSpinBox(),

            # Run PolarityJam block
            "label_rp": QLabel("Polarity-Jam execution:"),
            "param_button": QPushButton("Parameter File"),
            "param_file_loaded_indicator": QLabel(),
            "segment_button": QPushButton("Segment Image"),
            "run_button": QPushButton("Run PolarityJam"),

            # Junction labeling block
            "label_jc": QLabel("Junction classification:"),
            "label_jclass": QLabel("Junction Class"),
            "dropdown_labeling": QComboBox(),

            # Output block
            "label_output": QLabel("Output parameters:"),
            "output_path_label": QLabel("Output Path:"),
            "output_path": QPushButton("Select Output Path"),
            "output_file_prefix_label": QLabel("Output File Prefix:"),
            "output_file_prefix": QLineEdit(),
        }

        # Set minimum and maximum values for the QSpinBox widgets
        for channel in ["channel_junction", "channel_nucleus", "channel_organelle", "channel_expression_marker"]:
            self.widgets[channel].setMinimum(-1)
            self.widgets[channel].setMaximum(100)
            self.widgets[channel].setValue(-1)

        # Set size policy of the widgets
        for widget in self.widgets.values():
            widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # indicator for parameter file loaded
        arrow_path = pkg_resources.resource_filename('jat.ui.resources', 'arrow.svg')
        self.widgets["param_file_loaded_indicator"].setPixmap(QPixmap(arrow_path))
        self.widgets["param_file_loaded_indicator"].setVisible(False)

        # Add items to the dropdown menu
        self.widgets["dropdown_labeling"].addItems(
            ["none", "straight", "thick", "thick/reticular", "reticular", "fingers"]
        )

        # add connections
        self.widgets["param_button"].clicked.connect(self.load_parameter_file)
        self.widgets["segment_button"].clicked.connect(self.run_segmentation)
        self.widgets["run_button"].clicked.connect(self.run_polarityjam)

        # add connections for text changed
        self.widgets["channel_junction"].textChanged.connect(self.on_junction_text_changed)
        self.widgets["channel_nucleus"].textChanged.connect(self.on_nucleus_text_changed)
        self.widgets["channel_organelle"].textChanged.connect(self.on_organelle_text_changed)
        self.widgets["channel_expression_marker"].textChanged.connect(self.on_expression_marker_text_changed)
        self.widgets["output_path"].clicked.connect(self.select_output_path)
        self.widgets["output_file_prefix"].textChanged.connect(self.on_output_file_prefix_text_changed)

        # build layout
        self._build_layout()

    def on_junction_text_changed(self):
        self.params_image.channel_junction = int(self.widgets["channel_junction"].text())

    def on_nucleus_text_changed(self):
        self.params_image.channel_nucleus = int(self.widgets["channel_nucleus"].text())

    def on_organelle_text_changed(self):
        self.params_image.channel_organelle = int(self.widgets["channel_organelle"].text())

    def on_expression_marker_text_changed(self):
        self.params_image.channel_expression_marker = int(self.widgets["channel_expression_marker"].text())

    def _build_layout(self):
        # Create block-wise layout
        self.vbox_input = QVBoxLayout()
        self.vbox_output = QVBoxLayout()
        self.vbox_run_pjam = QVBoxLayout()
        self.vbox_junction_labeling = QVBoxLayout()

        # Input block
        self.vbox_input.addWidget(self.widgets["label_input"])
        for channel in ["channel_junction", "channel_nucleus", "channel_organelle", "channel_expression_marker"]:
            hbox = QHBoxLayout()
            hbox.addWidget(self.widgets[channel + "_label"])
            hbox.addWidget(self.widgets[channel])
            self.vbox_input.addLayout(hbox)

        # Output block
        self.vbox_output.addWidget(self.widgets["label_output"])
        hbox = QHBoxLayout()
        hbox.addWidget(self.widgets["output_path_label"])
        hbox.addWidget(self.widgets["output_path"])
        self.vbox_output.addLayout(hbox)
        hbox = QHBoxLayout()
        hbox.addWidget(self.widgets["output_file_prefix_label"])
        hbox.addWidget(self.widgets["output_file_prefix"])
        self.vbox_output.addLayout(hbox)

        # Run PolarityJam block
        self.vbox_run_pjam.addWidget(self.widgets["label_rp"])
        hbox = QHBoxLayout()
        hbox.addWidget(self.widgets["param_button"], 90)
        hbox.addWidget(self.widgets["param_file_loaded_indicator"], 10)
        self.vbox_run_pjam.addLayout(hbox)
        self.vbox_run_pjam.addWidget(self.widgets["segment_button"])
        self.vbox_run_pjam.addWidget(self.widgets["run_button"])

        # Junction labeling block
        self.vbox_junction_labeling.addWidget(self.widgets["label_jc"])
        hbox = QHBoxLayout()
        hbox.addWidget(self.widgets["label_jclass"])
        hbox.addWidget(self.widgets["dropdown_labeling"])
        self.vbox_junction_labeling.addLayout(hbox)

        # Add layouts to the overall layout
        self.layout.addLayout(self.vbox_input)
        self.layout.addLayout(self.vbox_output)
        self.layout.addLayout(self.vbox_run_pjam)
        self.layout.addLayout(self.vbox_junction_labeling)

    def select_output_path(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Path")
        if dir_path:  # if user didn't cancel the dialog
            self.output_path = dir_path
            self.widgets["output_path"].setText(f"{dir_path}")

    def on_output_file_prefix_text_changed(self):
        new_text = self.widgets["output_file_prefix"].text()
        self.output_path_prefix = new_text

    def segment_image(self):
        if self.params_seg is None:
            self.params_seg = SegmentationParameter(self.params_runtime.segmentation_algorithm)

        segmenter, _ = load_segmenter(self.params_runtime, self.params_seg)

        img1 = self.access_image()

        img_channels, _ = segmenter.prepare(img1, self.params_image)
        try:
            mask = segmenter.segment(img_channels)
        except Exception as e:
            # show error in pop up
            self.show_error_dialog(e)
            mask = None

        return mask

    def show_error_dialog(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("An error occurred")
        msg.setInformativeText(str(message))
        msg.setWindowTitle("Error")
        msg.exec_()

    def run_segmentation(self):
        # todo: asynchron call
        mask = self.segment_image()
        if mask is not None:
            self.add_mask_to_viewer(mask)

    def run_polarityjam(self):
        # todo: asynchron call
        # run polarityjam for the selected image
        collection = PropertiesCollection()
        extractor = Extractor(self.params_runtime)

        img = self.access_image()
        mask = self.access_mask()

        extractor.extract(img, self.params_image, mask, self.output_path_prefix, self.output_path, collection)

    def load_parameter_file(self):
        # Open a file dialog and load a YML file
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "YML Files (*.yml)")
        if file_path:
            print(f"Parameter file loaded: {file_path}")

        self.params_runtime = RuntimeParameter.from_yml(file_path)
        self.params_plot = PlotParameter.from_yml(file_path)
        self.params_seg = SegmentationParameter.from_yml(file_path)

        self.widgets["param_file_loaded_indicator"].setVisible(True)

    def access_image(self):
        image_data = None
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.image.image.Image):
                image_data = layer.data

        if image_data is None:
            raise ValueError("No image found in the viewer!")

        if image_data.shape[0] < min(image_data.shape[1], image_data.shape[2]):
            img = np.swapaxes(np.swapaxes(image_data, 0, 2), 0, 1)
        else:
            img = image_data

        return img

    def access_mask(self):
        mask = None
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.labels.labels.Labels):
                mask = layer.data

        if mask is None:
            raise ValueError("No mask found in the viewer!")

        return mask

    def add_mask_to_viewer(self, mask):
        self.viewer.add_labels(mask, name="PolarityJam Mask")
