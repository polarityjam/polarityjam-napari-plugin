from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGraphicsScene, QLabel, QLineEdit, QPushButton, QComboBox, \
    QSizePolicy, QHBoxLayout, QFileDialog

from polarityjam import Extractor, PropertiesCollection, load_segmenter
from polarityjam import RuntimeParameter, PlotParameter, SegmentationParameter, ImageParameter

import os
class JunctionAnnotationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.params_image = ImageParameter()
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
            "channel_junction": QLineEdit("-1"),
            "channel_nucleus": QLineEdit("-1"),
            "channel_organelle": QLineEdit("-1"),
            "channel_expression_marker": QLineEdit("-1"),

            # Run PolarityJam block
            "label_rp": QLabel("run PolarityJam on:"),
            "run_button": QPushButton("Run PolarityJam"),
            "label_jc": QLabel("Junction Class"),
            "dropdown_labeling": QComboBox(),
            "param_button": QPushButton("Parameter File"),

            # Output block
            "label_output": QLabel("Output parameters:"),
            "output_path_label": QLabel("Output Path:"),
            "output_path": QPushButton("Select Output Path"),
            "output_file_prefix_label": QLabel("Output File Prefix:"),
            "output_file_prefix": QLineEdit(),
        }

        # Set size policy of the widgets
        for widget in self.widgets.values():
            widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # Add items to the dropdown menu
        self.widgets["dropdown_labeling"].addItems(["Value 1", "Value 2", "Value 3", "Value 4", "Value 5"])

        # add connections
        self.widgets["param_button"].clicked.connect(self.load_parameter_file)
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
        self.params_image.channel_junction = int(self.widgets["channel_nucleus"].text())

    def on_organelle_text_changed(self):
        self.params_image.channel_junction = int(self.widgets["channel_organelle"].text())

    def on_expression_marker_text_changed(self):
        self.params_image.channel_junction = int(self.widgets["channel_expression_marker"].text())

    def _build_layout(self):
        # Create block-wise layout
        self.vbox_input = QVBoxLayout()
        self.vbox_output = QVBoxLayout()
        self.vbox_run_pjam = QVBoxLayout()
        self.hbox_junction_labeling = QHBoxLayout()

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
        self.vbox_run_pjam.addWidget(self.widgets["param_button"])
        self.vbox_run_pjam.addWidget(self.widgets["run_button"])

        # Junction labeling block
        self.hbox_junction_labeling.addWidget(self.widgets["label_jc"])
        self.hbox_junction_labeling.addWidget(self.widgets["dropdown_labeling"])

        # Add layouts to the overall layout
        self.layout.addLayout(self.vbox_input)
        self.layout.addLayout(self.vbox_output)
        self.layout.addLayout(self.vbox_run_pjam)
        self.layout.addLayout(self.hbox_junction_labeling)

    def select_output_path(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Path")
        if dir_path:  # if user didn't cancel the dialog
            self.output_path = dir_path
            self.widgets["output_path"].setText(f"Output Path: {dir_path}")

    def on_output_file_prefix_text_changed(self):
        new_text = self.widgets["output_file_prefix"].text()
        self.output_path_prefix = new_text

    def segment_image(self):
        if self.params_seg is None:
            self.params_seg = SegmentationParameter(self.params_runtime.segmentation_algorithm)

        segmenter, _ = load_segmenter(self.params_runtime, self.params_seg)

        img_channels, _ = segmenter.prepare(img1, params_image1)
        mask = segmenter.segment(img_channels, input_file1)

        return mask

    def run_polarityjam(self):
        # run polarityjam for the selected image
        mask = self.segment_image()

        collection = PropertiesCollection()
        extractor = Extractor(self.params_runtime)
        extractor.extract(img1, params_image1, mask, output_file_prefix1, output_path, collection)

    def load_parameter_file(self):
        # Open a file dialog and load a YML file
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "YML Files (*.yml)")
        if file_path:
            print(f"Parameter file loaded: {file_path}")

        self.params_runtime = RuntimeParameter.from_yml(file_path)
        self.params_plot = PlotParameter.from_yml(file_path)
        self.params_seg = SegmentationParameter.from_yml(file_path)
