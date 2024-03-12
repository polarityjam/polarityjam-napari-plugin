import os

import napari
import numpy as np
import pkg_resources
from PyQt5.QtCore import QRunnable, pyqtSlot, pyqtSignal, QObject, QTimer
from PyQt5.QtCore import QThreadPool
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGraphicsScene, QLabel, QLineEdit, QPushButton, QComboBox, \
    QSizePolicy, QHBoxLayout, QFileDialog, QMessageBox, QSpinBox
from polarityjam import Extractor, PropertiesCollection, load_segmenter
from polarityjam import RuntimeParameter, PlotParameter, SegmentationParameter, ImageParameter


class WorkerSignalsSegmentation(QObject):
    finished = pyqtSignal(np.ndarray)  # Signal that will be emitted when the task finishes
    error = pyqtSignal(Exception)


class WorkerSignalsExtraction(QObject):
    features_extracted = pyqtSignal(PropertiesCollection)  # Signal that will be emitted when features are extracted
    error = pyqtSignal(Exception)


class RunSegmentationTask(QRunnable):
    def __init__(self, img, params_seg, params_runtime, params_image):
        super().__init__()
        self.img = img
        self.params_seg = params_seg
        self.params_runtime = params_runtime
        self.params_image = params_image
        self.signals = WorkerSignalsSegmentation()

    @pyqtSlot()
    def run(self):
        try:
            mask = self.segment_image()
        except Exception as e:
            self.signals.error.emit(e)

        self.signals.finished.emit(mask)

    def segment_image(self):
        if self.params_seg is None:
            self.params_seg = SegmentationParameter(self.params_runtime.segmentation_algorithm)

        segmenter, _ = load_segmenter(self.params_runtime, self.params_seg)

        img_channels, _ = segmenter.prepare(self.img, self.params_image)
        try:
            mask = segmenter.segment(img_channels)
        except Exception as e:
            self.signals.error.emit(e)
            mask = None

        return mask


class RunPolarityJamTask(QRunnable):
    def __init__(self, img, mask, params_image, output_path_prefix, output_path):
        super().__init__()
        self.signals = WorkerSignalsExtraction()
        self.img = img
        self.mask = mask
        self.params_image = params_image
        self.output_path_prefix = output_path_prefix
        self.output_path = output_path

    @pyqtSlot()
    def run(self):
        try:
            collection = self.extract_features()
        except Exception as e:
            self.signals.error.emit(e)
            collection = None
        self.signals.features_extracted.emit(collection)

    def extract_features(self):
        collection = PropertiesCollection()
        extractor = Extractor(self.params_runtime)

        extractor.extract(self.img, self.params_image, self.mask, self.output_path_prefix, self.output_path, collection)

        return collection


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

        self.collection = None

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.scene = QGraphicsScene()

        self.loading_timer_feature_extraction = QTimer()
        self.loading_timer_segmentation = QTimer()

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
            "feature_extraction_indicator": QLabel(),
            "segmentation_indicator": QLabel(),
            "segment_button": QPushButton("Segment Image"),
            "run_button": QPushButton("Run PolarityJam"),

            # Junction labeling block
            "label_jc": QLabel("Junction classification:"),
            "label_jclass": QLabel("Junction Class"),
            "dropdown_labeling": QComboBox(),
            "previous_button": QPushButton("Previous"),
            "next_button": QPushButton("Next"),

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

        loading_path = pkg_resources.resource_filename('jat.ui.resources', 'loading.svg')
        self.widgets["feature_extraction_indicator"].setPixmap(QPixmap(loading_path))
        self.widgets["feature_extraction_indicator"].setVisible(False)

        self.widgets["segmentation_indicator"].setPixmap(QPixmap(loading_path))
        self.widgets["segmentation_indicator"].setVisible(False)

        # Add items to the dropdown menu
        self.widgets["dropdown_labeling"].addItems(
            ["none", "straight", "thick", "thick/reticular", "reticular", "fingers"]
        )

        # add connections
        self.widgets["param_button"].clicked.connect(self.load_parameter_file)
        self.widgets["segment_button"].clicked.connect(self.run_segmentation)
        self.widgets["run_button"].clicked.connect(self.run_polarityjam)
        self.widgets["previous_button"].clicked.connect(self.previous_button_clicked)
        self.widgets["next_button"].clicked.connect(self.next_button_clicked)

        # add connections for text changed
        self.widgets["channel_junction"].textChanged.connect(self.on_junction_text_changed)
        self.widgets["channel_nucleus"].textChanged.connect(self.on_nucleus_text_changed)
        self.widgets["channel_organelle"].textChanged.connect(self.on_organelle_text_changed)
        self.widgets["channel_expression_marker"].textChanged.connect(self.on_expression_marker_text_changed)
        self.widgets["output_path"].clicked.connect(self.select_output_path)
        self.widgets["output_file_prefix"].textChanged.connect(self.on_output_file_prefix_text_changed)

        # timer connection
        self.loading_timer_feature_extraction.timeout.connect(self.change_loading_image_feature_extraction)
        self.loading_timer_segmentation.timeout.connect(self.change_loading_image_segmentation)
        self.segmentation_indicator_state = True
        self.feature_extraction_indicator_state = True
        # build layout
        self._build_layout()

    def previous_button_clicked(self):
        # This function will be executed when the "Previous" button is clicked
        if self.collection is None:
            # should run polarityjam first
            return

    def next_button_clicked(self):
        # This function will be executed when the "Next" button is clicked
        if self.collection is None:
            # should run polarityjam first
            return

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
            hbox.addWidget(self.widgets[channel + "_label"], 90)
            hbox.addWidget(self.widgets[channel], 10)
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
        hbox = QHBoxLayout()
        hbox.addWidget(self.widgets["segment_button"], 90)
        hbox.addWidget(self.widgets["segmentation_indicator"], 10)
        self.vbox_run_pjam.addLayout(hbox)
        hbox = QHBoxLayout()
        hbox.addWidget(self.widgets["run_button"], 90)
        hbox.addWidget(self.widgets["feature_extraction_indicator"], 10)
        self.vbox_run_pjam.addLayout(hbox)

        # Junction labeling block
        self.vbox_junction_labeling.addWidget(self.widgets["label_jc"])
        hbox = QHBoxLayout()
        hbox.addWidget(self.widgets["label_jclass"])
        hbox.addWidget(self.widgets["dropdown_labeling"])
        self.vbox_junction_labeling.addLayout(hbox)
        hbox = QHBoxLayout()
        hbox.addWidget(self.widgets["previous_button"])
        hbox.addWidget(self.widgets["next_button"])
        self.vbox_junction_labeling.addLayout(hbox)

        # Add layouts to the overall layout
        self.layout.addLayout(self.vbox_input)
        self.layout.addLayout(self.vbox_output)
        self.layout.addLayout(self.vbox_run_pjam)
        self.layout.addLayout(self.vbox_junction_labeling)

    def change_loading_image_feature_extraction(self):
        loading_path = pkg_resources.resource_filename('jat.ui.resources', 'loading.svg')
        loading_v_path = pkg_resources.resource_filename('jat.ui.resources', 'loading_v.svg')

        if self.feature_extraction_indicator_state:
            self.widgets["feature_extraction_indicator"].setPixmap(QPixmap(loading_v_path))
        else:
            self.widgets["feature_extraction_indicator"].setPixmap(QPixmap(loading_path))

        self.feature_extraction_indicator_state = not self.feature_extraction_indicator_state

    def change_loading_image_segmentation(self):
        loading_path = pkg_resources.resource_filename('jat.ui.resources', 'loading.svg')
        loading_v_path = pkg_resources.resource_filename('jat.ui.resources', 'loading_v.svg')

        if self.segmentation_indicator_state:
            self.widgets["segmentation_indicator"].setPixmap(QPixmap(loading_v_path))
        else:
            self.widgets["segmentation_indicator"].setPixmap(QPixmap(loading_path))

        self.segmentation_indicator_state = not self.segmentation_indicator_state

    def select_output_path(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Path")
        if dir_path:  # if user didn't cancel the dialog
            self.output_path = dir_path
            self.widgets["output_path"].setText(f"{dir_path}")

    def on_output_file_prefix_text_changed(self):
        new_text = self.widgets["output_file_prefix"].text()
        self.output_path_prefix = new_text

    def run_segmentation(self):
        # Create a QThreadPool instance
        thread_pool = QThreadPool().globalInstance()

        # Create a RunSegmentationTask instance
        img = self.access_image()
        task = RunSegmentationTask(img, self.params_seg, self.params_runtime, self.params_image)

        # Connect the error signal to the handle_error method
        task.signals.error.connect(self.handle_error)

        # connect the finished signal to the handle_segmentation_result method
        task.signals.finished.connect(self.handle_segmentation_result)

        # Start the task
        thread_pool.start(task)

        # Set the visibility of segmentation_indicator to True
        self.widgets["segmentation_indicator"].setVisible(True)

        # start the loading timer
        self.loading_timer_segmentation.start(2000)

    def handle_error(self, e):
        # This function will be called when an error occurs in the RunSegmentationTask
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("An error occurred")
        msg.setInformativeText(str(e))
        msg.setWindowTitle("Error")
        msg.exec_()

    def handle_segmentation_result(self, mask):
        # This function will be called when the RunSegmentationTask finishes
        # The mask parameter will contain the result of the segment_image function

        # stop the loading timer
        self.loading_timer_segmentation.stop()

        if mask is not None:
            self.add_mask_to_viewer(mask)

            # Change the image of feature_extraction_indicator to arrow.svg
            arrow_path = pkg_resources.resource_filename('jat.ui.resources', 'arrow.svg')
            self.widgets["segmentation_indicator"].setPixmap(QPixmap(arrow_path))
        else:
            # Set the visibility of segmentation_indicator to False
            self.widgets["segmentation_indicator"].setVisible(False)

    def run_polarityjam(self):
        # Create a QThreadPool instance
        thread_pool = QThreadPool()

        # Create a RunPolarityJamTask instance
        img = self.access_image()
        mask = self.access_mask()
        task = RunPolarityJamTask(self, img, mask, self.params_image, self.output_path_prefix, self.output_path)

        # connect the features_extracted signal to the handle_features_extraction_result method
        task.signals.features_extracted.connect(self.handle_features_extraction_result)

        # Connect the error signal to the handle_error method
        task.signals.error.connect(self.handle_error)

        # Start the task
        thread_pool.start(task)

        # Set the visibility of feature_extraction_indicator to True
        self.widgets["feature_extraction_indicator"].setVisible(True)

        # start the loading timer
        self.loading_timer_feature_extraction.start(2000)

    def handle_features_extraction_result(self, collection):
        # This function will be called when the RunPolarityJamTask finishes
        # The collection parameter will contain the result of the extract_features function
        self.collection = collection

        # stop the loading timer
        self.loading_timer_feature_extraction.stop()

        # Change the image of feature_extraction_indicator to arrow.svg
        arrow_path = pkg_resources.resource_filename('jat.ui.resources', 'arrow.svg')
        self.widgets["feature_extraction_indicator"].setPixmap(QPixmap(arrow_path))

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
