"""Main module for the PolarityJam Napari plugin."""
import glob
import json
import os
import webbrowser
from pathlib import Path

import napari
import numpy as np
import pandas as pd
from importlib.resources import files
import requests
from PyQt5.QtCore import QThreadPool, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGraphicsScene,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from skimage.morphology import binary_dilation

from polarityjam import (
    ImageParameter,
    PlotParameter,
    RuntimeParameter,
    SegmentationParameter,
)
from polarityjam.napari_plugin.model.tasks import (
    PlotFeaturesTask,
    RunPolarityJamTask,
    RunSegmentationTask,
)


def fetch_urls():
    """Fetch the urls from the resource-links.json file in the repository."""
    try:
        response = requests.get(
            "https://raw.githubusercontent.com/polarityjam/polarityjam/doc/docs/resource-links.json"
        )
        urls = json.loads(response.text)
    except BaseException:  # noqa: B036
        # default urls point to the github repository
        return {
            "WEB_URL_DOCS": "https://github.com/polarityjam/polarityjam",
            "WEB_URL_APP": "https://github.com/polarityjam/polarityjam",
            "WEB_URL_ARTICLE": "https://github.com/polarityjam/polarityjam",
        }
    return urls


class PjamNapariWidget(QWidget):
    """Main widget for the PolarityJam Napari plugin."""

    urls = fetch_urls()
    WEB_URL_DOCS = urls["WEB_URL_DOCS"]
    WEB_URL_APP = urls["WEB_URL_APP"]
    WEB_URL_ARTICLE = urls["WEB_URL_ARTICLE"]

    resource_files = files("polarityjam.napari_plugin.ui.resources")

    PATH_TO_ARROW = str(resource_files.joinpath("arrow.svg"))
    PATH_TO_LOADING = str(resource_files.joinpath("loading.svg"))
    PATH_TO_LOADING_V = str(resource_files.joinpath("loading_v.svg"))

    def __init__(self, napari_viewer):
        """Initialize the widget."""
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
        self.loading_timer_plot = QTimer()

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
            # Extract options block
            "label_extract": QLabel("Extract options:"),
            "extract_morphology": QCheckBox("Morphology"),
            "extract_polarity": QCheckBox("Polarity"),
            "extract_intensity": QCheckBox("Marker"),
            "extract_topology": QCheckBox("Topology"),
            # Run PolarityJam block
            "label_rp": QLabel("Polarity-Jam execution:"),
            "param_button": QPushButton("Parameter File"),
            "param_file_loaded_indicator": QLabel(),
            "feature_extraction_indicator": QLabel(),
            "segmentation_indicator": QLabel(),
            "plot_indicator": QLabel(),
            "segment_button": QPushButton("Segment Image"),
            "run_button": QPushButton("Run PolarityJam"),
            "plot_button": QPushButton("Plot Features"),
            "label_loadback": QLabel("Load plot:"),
            "dropdown_loadback": QComboBox(),
            "reset_button": QPushButton("Reset plots"),
            # Junction labeling block
            "label_jc": QLabel("Junction classification:"),
            "label_jclass": QLabel("Junction Class"),
            "dropdown_labeling": QComboBox(),
            "previous_button": QPushButton("Previous"),
            "next_button": QPushButton("Next"),
            "thickness_label": QLabel("Thickness:"),
            "thickness": QSpinBox(),
            "save_button": QPushButton("Save"),
            "save_indicator": QLabel(),
            # Output block
            "label_output": QLabel("Output parameters:"),
            "output_path_label": QLabel("Output Path:"),
            "output_path": QPushButton("Select Output Path"),
            "output_file_prefix_label": QLabel("Output File Prefix:"),
            "output_file_prefix": QLineEdit(),
            # help block
            "docs_button": QPushButton("Docs"),
            "web_button": QPushButton("App"),
            "article_button": QPushButton("Article"),
        }

        # Set default checked state for the checkboxes
        for checkbox in [
            "extract_morphology",
            "extract_polarity",
            "extract_intensity",
            "extract_topology",
        ]:
            self.widgets[checkbox].setChecked(True)

        # Set minimum and maximum values for the QSpinBox widgets
        for channel in [
            "channel_junction",
            "channel_nucleus",
            "channel_organelle",
            "channel_expression_marker",
        ]:
            self.widgets[channel].setMinimum(-1)
            self.widgets[channel].setMaximum(100)
            self.widgets[channel].setValue(-1)

        # default value for thickness
        self.widgets["thickness"].setMinimum(1)
        self.widgets["thickness"].setMaximum(10)
        self.widgets["thickness"].setValue(3)

        # Set size policy of the widgets
        for widget in self.widgets.values():
            widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # indicator for parameter file loaded
        self.widgets["param_file_loaded_indicator"].setPixmap(
            QPixmap(self.PATH_TO_ARROW)
        )
        self.widgets["param_file_loaded_indicator"].setVisible(False)

        # indicator for saving the junction label dataset
        self.widgets["save_indicator"].setPixmap(QPixmap(self.PATH_TO_ARROW))
        self.widgets["save_indicator"].setVisible(False)

        self.widgets["feature_extraction_indicator"].setPixmap(
            QPixmap(self.PATH_TO_LOADING)
        )
        self.widgets["feature_extraction_indicator"].setVisible(False)

        self.widgets["segmentation_indicator"].setPixmap(QPixmap(self.PATH_TO_LOADING))
        self.widgets["segmentation_indicator"].setVisible(False)

        self.widgets["plot_indicator"].setPixmap(QPixmap(self.PATH_TO_LOADING))
        self.widgets["plot_indicator"].setVisible(False)

        # Add items to the dropdown menu
        self.widgets["dropdown_labeling"].addItems(
            ["none", "straight", "thick", "thick/reticular", "reticular", "fingers"]
        )

        # add connections
        self.widgets["param_button"].clicked.connect(self.load_parameter_file)
        self.widgets["segment_button"].clicked.connect(self.run_segmentation)
        self.widgets["run_button"].clicked.connect(self.run_polarityjam)
        self.widgets["plot_button"].clicked.connect(self.run_plot)
        self.widgets["dropdown_loadback"].currentIndexChanged.connect(
            self.on_dropdown_loadback_changed
        )
        self.widgets["reset_button"].clicked.connect(self.reset_plots)
        self.widgets["previous_button"].clicked.connect(self.previous_button_clicked)
        self.widgets["next_button"].clicked.connect(self.next_button_clicked)
        self.widgets["dropdown_labeling"].currentIndexChanged.connect(
            self.on_dropdown_labeling_changed
        )
        self.widgets["save_button"].clicked.connect(self.save_dataset)
        self.widgets["docs_button"].clicked.connect(
            lambda: webbrowser.open(self.WEB_URL_DOCS)
        )
        self.widgets["web_button"].clicked.connect(
            lambda: webbrowser.open(self.WEB_URL_APP)
        )
        self.widgets["article_button"].clicked.connect(
            lambda: webbrowser.open(self.WEB_URL_ARTICLE)
        )

        # add connections for text changed
        self.widgets["channel_junction"].textChanged.connect(
            self.on_junction_text_changed
        )
        self.widgets["channel_nucleus"].textChanged.connect(
            self.on_nucleus_text_changed
        )
        self.widgets["channel_organelle"].textChanged.connect(
            self.on_organelle_text_changed
        )
        self.widgets["channel_expression_marker"].textChanged.connect(
            self.on_expression_marker_text_changed
        )
        self.widgets["output_path"].clicked.connect(self.select_output_path)
        self.widgets["output_file_prefix"].textChanged.connect(
            self.on_output_file_prefix_text_changed
        )
        self.widgets["thickness"].textChanged.connect(self.on_thickness_text_changed)

        # timer connection
        self.loading_timer_feature_extraction.timeout.connect(
            self.change_loading_image_feature_extraction
        )
        self.loading_timer_segmentation.timeout.connect(
            self.change_loading_image_segmentation
        )
        self.loading_timer_plot.timeout.connect(self.change_loading_image_plot)
        self.segmentation_indicator_state = True
        self.feature_extraction_indicator_state = True
        self.plot_indicator_state = True

        # other initializations
        self.cur_index = -1
        self.overlap = 1
        self.biomed_img = None
        self.neighbors_combination_list = []
        self._cells_layer_list = []
        self.junction_label = pd.DataFrame(
            columns=["label_1", "label_2", "junction_class"]
        )
        self._param_loaded = False

        # build layout
        self._build_layout()

    def list_png_files(self):
        """List all the png files in the output directory."""
        search_path = os.path.join(self.output_path, self.output_path_prefix + "*.png")
        r_list = glob.glob(search_path)

        # return only the file names without prefix
        r_list = [
            os.path.basename(f).replace(self.output_path_prefix + "_", "")
            for f in r_list
        ]
        r_list = [f.replace(".png", "") for f in r_list]
        return r_list

    def on_thickness_text_changed(self):
        """Update the overlap value from the thickness text box."""
        self.overlap = int(self.widgets["thickness"].text())

    def save_dataset(self):
        """Save the junction label dataset to a csv file."""
        # Specify the path and name of the file to save the dataset
        file_path = Path(self.output_path).joinpath(
            f"{self.output_path_prefix}_junction_label.csv"
        )

        if self.neighbors_combination_list == []:
            # should run polarityjam first
            return

        self.junction_label.to_csv(file_path, index=False)

        # toggle visibility of the save indicator
        self.widgets["save_indicator"].setVisible(True)

    def save_feature_and_collection(self, collection):
        """Save the feature dataset to a csv file."""
        # Specify the path and name of the file to save the dataset
        file_path = Path(self.output_path).joinpath(
            f"{self.output_path_prefix}_features.csv"
        )
        collection_path = Path(self.output_path).joinpath(
            f"{self.output_path_prefix}_collection.pkl"
        )

        if self.collection is None or len(self.collection) == 0:
            return

        collection.save(collection_path)
        collection.dataset.to_csv(file_path, index=False)

    def on_dropdown_loadback_changed(self):
        """Load the selected plot image into napari."""
        # do nothing if no collection is available
        if self.collection is None or len(self.collection) == 0:
            return

        # get the selected item from the dropdown menu
        selected_item = self.widgets["dropdown_loadback"].currentText()

        # build path to the file
        file_path = Path(self.output_path).joinpath(
            f"{self.output_path_prefix}_{selected_item}.png"
        )

        # load the plot_image into napari
        self.viewer.open(str(file_path))

    def on_dropdown_labeling_changed(self):
        """Update the junction class in the junction_label dataframe."""
        # do nothing if no collection is available
        if self.collection is None or len(self.collection) == 0:
            return

        # get the selected item from the dropdown menu
        selected_item = self.widgets["dropdown_labeling"].currentText()

        label, neighbor = self.neighbors_combination_list[self.cur_index]

        # change the junction class in the junction_label dataframe
        self.junction_label.loc[
            (self.junction_label["label_1"] == label)
            & (self.junction_label["label_2"] == neighbor),
            "junction_class",
        ] = selected_item

    def previous_button_clicked(self):
        """Show the previous junction in the list."""
        # disable to disallow double clicking
        self.widgets["previous_button"].setEnabled(False)

        if len(self.neighbors_combination_list) == 0:
            # should run polarityjam first
            self.widgets["previous_button"].setEnabled(True)
            return

        # decrease current index if possible
        if self.cur_index > 0:
            self.cur_index -= 1
        else:
            # enable the button
            self.widgets["previous_button"].setEnabled(True)
            return

        # reset drop down menu to that of the junction_label dataframe
        self._reset_junction_box()

        self._show_single_junction()

        # enable the button
        self.widgets["previous_button"].setEnabled(True)

    def _get_all_neighbor_combinations(self):
        if not (self.collection is None or len(self.collection) == 0):
            self.biomed_img = self._get_biomed_img()

            # get all unique neighbor combinations
            for (
                label
            ) in self.biomed_img.segmentation.segmentation_mask_connected.get_labels():
                neighbors = list(
                    self.biomed_img.segmentation.neighborhood_graph_connected.neighbors(
                        label
                    )
                )

                for neighbor in neighbors:
                    if label < neighbor:
                        self.neighbors_combination_list.append((label, neighbor))
                        # add default value to the junction_label dataframe
                        self.junction_label = self.junction_label.append(
                            {
                                "label_1": label,
                                "label_2": neighbor,
                                "junction_class": "none",
                            },
                            ignore_index=True,
                        )

    def next_button_clicked(self):
        """Show the next junction in the list."""
        # disable to disallow double clicking
        self.widgets["next_button"].setEnabled(False)

        if len(self.neighbors_combination_list) == 0:
            # should run polarityjam first
            self.widgets["next_button"].setEnabled(True)
            return

        # increase index if possible
        if self.cur_index < len(self.neighbors_combination_list) - 1:
            self.cur_index += 1
        else:
            self.widgets["next_button"].setEnabled(True)
            return

        # reset drop down menu to that of the junction_label dataframe
        self._reset_junction_box()

        self._show_single_junction()

        # enable the button
        self.widgets["next_button"].setEnabled(True)

    def _reset_junction_box(self):
        label, neighbor = self.neighbors_combination_list[self.cur_index]
        junction_class = self.junction_label.loc[
            (self.junction_label["label_1"] == label)
            & (self.junction_label["label_2"] == neighbor),
            "junction_class",
        ].values[0]
        self.widgets["dropdown_labeling"].setCurrentText(junction_class)

    def _show_single_junction(self):
        label, neighbor = self.neighbors_combination_list[self.cur_index]
        sc_mask = self.biomed_img.segmentation.segmentation_mask_connected.get_single_instance_mask(
            label
        )
        sc_mask_n = self.biomed_img.segmentation.segmentation_mask_connected.get_single_instance_mask(
            neighbor
        )

        # dilate the masks and calculate overlap
        combined_mask = self.get_sc_junction_mask(sc_mask, sc_mask_n)

        # add the combined mask to the viewer
        cur_mask_name = "sc_mask%d_%d" % (label, neighbor)
        self.add_mask_to_viewer(combined_mask.data, cur_mask_name)
        self._cells_layer_list.append(cur_mask_name)

        # remove the previous mask if exists
        if len(self._cells_layer_list) > 1:
            _name = self._cells_layer_list.pop(0)  # holds previous mask name
            if _name in self.viewer.layers:
                # remove from layers view
                self.viewer.layers.pop(_name)

        # disable the view on the segmentation mask
        self.viewer.layers["PolarityJam Mask"].visible = False

    def _get_biomed_img(self):
        cur_name = self.collection.dataset.at[1, "filename"]
        biomed_img = self.collection.get_image_by_img_name(cur_name)
        return biomed_img

    def get_sc_junction_mask(self, sc_mask, sc_mask_n):
        """Get the junction mask for the given supercell masks."""
        sc_mask_d = binary_dilation(sc_mask.data)
        sc_mask_n_d = binary_dilation(sc_mask_n.data)
        for _ in range(self.overlap - 1):
            sc_mask_d = binary_dilation(sc_mask_d.data)
            sc_mask_n_d = binary_dilation(sc_mask_n_d.data)
        overlap = np.where(np.logical_and(sc_mask_d, sc_mask_n_d))
        combined_mask = sc_mask.disjoin(sc_mask_n).to_instance_mask(1)

        # add overlap mask on top
        combined_mask.data[overlap] = 2

        return combined_mask

    def on_junction_text_changed(self):
        """Update the junction channel from the text box."""
        self.params_image.channel_junction = int(
            self.widgets["channel_junction"].text()
        )

    def on_nucleus_text_changed(self):
        """Update the nucleus channel from the text box."""
        self.params_image.channel_nucleus = int(self.widgets["channel_nucleus"].text())

    def on_organelle_text_changed(self):
        """Update the organelle channel from the text box."""
        self.params_image.channel_organelle = int(
            self.widgets["channel_organelle"].text()
        )

    def on_expression_marker_text_changed(self):
        """Update the expression marker channel from the text box."""
        self.params_image.channel_expression_marker = int(
            self.widgets["channel_expression_marker"].text()
        )

    def _build_layout(self):
        # Create block-wise layout
        self.vbox_input = QVBoxLayout()
        self.vbox_extraction = QVBoxLayout()
        self.vbox_output = QVBoxLayout()
        self.vbox_run_pjam = QVBoxLayout()
        self.vbox_junction_labeling = QVBoxLayout()
        self.vbox_help = QVBoxLayout()

        # Input block
        self.vbox_input.addWidget(self.widgets["label_input"])
        for channel in [
            "channel_junction",
            "channel_nucleus",
            "channel_organelle",
            "channel_expression_marker",
        ]:
            hbox = QHBoxLayout()
            hbox.addWidget(self.widgets[channel + "_label"], 90)
            hbox.addWidget(self.widgets[channel], 10)
            self.vbox_input.addLayout(hbox)

        # Extract options block
        self.vbox_extraction.addWidget(self.widgets["label_extract"])
        hbox = QHBoxLayout()
        hbox.addWidget(self.widgets["extract_morphology"])
        hbox.addWidget(self.widgets["extract_polarity"])
        self.vbox_extraction.addLayout(hbox)
        hbox = QHBoxLayout()
        hbox.addWidget(self.widgets["extract_intensity"])
        hbox.addWidget(self.widgets["extract_topology"])
        self.vbox_extraction.addLayout(hbox)

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
        hbox = QHBoxLayout()
        hbox.addWidget(self.widgets["plot_button"], 90)
        hbox.addWidget(self.widgets["plot_indicator"], 10)
        self.vbox_run_pjam.addLayout(hbox)
        hbox = QHBoxLayout()
        hbox.addWidget(self.widgets["label_loadback"])
        hbox.addWidget(self.widgets["dropdown_loadback"])
        hbox.addWidget(self.widgets["reset_button"])
        self.vbox_run_pjam.addLayout(hbox)

        # Junction labeling block - disabled for now
        # self.vbox_junction_labeling.addWidget(self.widgets["label_jc"])
        # hbox = QHBoxLayout()
        # hbox.addWidget(self.widgets["label_jclass"])
        # hbox.addWidget(self.widgets["dropdown_labeling"])
        # self.vbox_junction_labeling.addLayout(hbox)
        # hbox = QHBoxLayout()
        # hbox.addWidget(self.widgets["previous_button"])
        # hbox.addWidget(self.widgets["next_button"])
        # self.vbox_junction_labeling.addLayout(hbox)
        # hbox = QHBoxLayout()
        # hbox.addWidget(self.widgets["thickness_label"])
        # hbox.addWidget(self.widgets["thickness"])
        # self.vbox_junction_labeling.addLayout(hbox)
        # hbox = QHBoxLayout()
        # hbox.addWidget(self.widgets["save_button"], 90)
        # hbox.addWidget(self.widgets["save_indicator"], 10)
        # self.vbox_junction_labeling.addLayout(hbox)

        # help block
        hbox = QHBoxLayout()
        hbox.addWidget(self.widgets["docs_button"])
        hbox.addWidget(self.widgets["web_button"])
        hbox.addWidget(self.widgets["article_button"])
        self.vbox_help.addLayout(hbox)

        # Add layouts to the overall layout
        self.layout.addLayout(self.vbox_input)
        self.layout.addLayout(self.vbox_extraction)
        self.layout.addLayout(self.vbox_output)
        self.layout.addLayout(self.vbox_run_pjam)
        self.layout.addLayout(self.vbox_junction_labeling)
        self.layout.addLayout(self.vbox_help)

    def change_loading_image_feature_extraction(self):
        """Change the loading image for feature extraction."""
        if self.feature_extraction_indicator_state:
            self.widgets["feature_extraction_indicator"].setPixmap(
                QPixmap(self.PATH_TO_LOADING_V)
            )
        else:
            self.widgets["feature_extraction_indicator"].setPixmap(
                QPixmap(self.PATH_TO_LOADING)
            )

        self.feature_extraction_indicator_state = (
            not self.feature_extraction_indicator_state
        )

    def change_loading_image_segmentation(self):
        """Change the loading image for segmentation."""
        if self.segmentation_indicator_state:
            self.widgets["segmentation_indicator"].setPixmap(
                QPixmap(self.PATH_TO_LOADING_V)
            )
        else:
            self.widgets["segmentation_indicator"].setPixmap(
                QPixmap(self.PATH_TO_LOADING)
            )

        self.segmentation_indicator_state = not self.segmentation_indicator_state

    def change_loading_image_plot(self):
        """Change the loading image for plotting."""
        if self.plot_indicator_state:
            self.widgets["plot_indicator"].setPixmap(QPixmap(self.PATH_TO_LOADING_V))
        else:
            self.widgets["plot_indicator"].setPixmap(QPixmap(self.PATH_TO_LOADING))

        self.plot_indicator_state = not self.plot_indicator_state

    def reset_plots(self):
        """Remove all the plots from the viewer."""
        r_list = self.list_png_files()
        r_list = [self.output_path_prefix + "_" + f for f in r_list]

        for layer in self.viewer.layers:
            # remove if the layer is a plot
            if layer.name in r_list:
                self.viewer.layers.remove(layer)

    def select_output_path(self):
        """Select the output path for the plots."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Path")
        if dir_path:  # if user didn't cancel the dialog
            self.output_path = dir_path
            self.widgets["output_path"].setText(f"{dir_path}")

    def on_output_file_prefix_text_changed(self):
        """Update the output file prefix from the text box."""
        new_text = self.widgets["output_file_prefix"].text()
        self.output_path_prefix = new_text

    def run_segmentation(self):
        """Run the segmentation task."""
        # Disable the button
        self.widgets["segment_button"].setEnabled(False)

        # Create a QThreadPool instance
        thread_pool = QThreadPool().globalInstance()

        # Create a RunSegmentationTask instance
        img, img_path = self.access_image("segmentation")

        if img is None:
            return

        if (
            self.params_image.channel_junction == -1
            and self.params_image.channel_nucleus == -1
            and self.params_image.channel_organelle == -1
            and self.params_image.channel_expression_marker == -1
        ):
            self.show_message(
                "No input channels provided!",
                "Please provide at least one input channel.",
                "segmentation",
            )
            return

        if not self._param_loaded:
            self.show_message(
                "No parameter file loaded!",
                "Please load a parameter file first.",
                "segmentation",
            )
            return

        task = RunSegmentationTask(
            img, self.params_seg, self.params_runtime, self.params_image, img_path
        )

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

    def handle_error(self, t):
        """Handle any error signal."""
        e, task_type = t

        # This function will be called when an error occurs in the RunSegmentationTask
        # enable the buttons again
        self.widgets["segment_button"].setEnabled(True)
        self.widgets["run_button"].setEnabled(True)
        self.widgets["plot_button"].setEnabled(True)

        # reset the loading timer
        self.loading_timer_feature_extraction.stop()
        self.loading_timer_segmentation.stop()
        self.loading_timer_plot.stop()

        # disable the indicator for the corresponding task
        if task_type == "Segmentation":
            self.widgets["segmentation_indicator"].setVisible(False)
        elif task_type == "Extraction":
            self.widgets["feature_extraction_indicator"].setVisible(False)
        elif task_type == "Plot":
            self.widgets["plot_indicator"].setVisible(False)

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("An error occurred")
        msg.setInformativeText(str(e))
        msg.setWindowTitle("Error")
        msg.exec_()

    def show_message(self, text, inf_text, call_class=None):
        """Show a message box with the given text and informative text."""
        self.widgets["segment_button"].setEnabled(True)
        self.widgets["run_button"].setEnabled(True)
        self.widgets["plot_button"].setEnabled(True)

        # disable timer and change the loading image
        if call_class == "segmentation":
            self.loading_timer_segmentation.stop()
            self.widgets["segmentation_indicator"].setVisible(False)
        elif call_class == "feature":
            self.loading_timer_feature_extraction.stop()
            self.widgets["feature_extraction_indicator"].setVisible(False)

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(text)
        msg.setInformativeText(inf_text)
        msg.setWindowTitle("Information")
        msg.exec_()

    def run_plot(self):
        """Run the PlotFeaturesTask."""
        self.widgets["plot_button"].setEnabled(False)

        if self.collection is None or len(self.collection) == 0:
            self.show_message("No features extracted!", "Please run PolarityJam first.")
            return

        print(self.params_plot.graphics_output_format)

        if "png" not in self.params_plot.graphics_output_format:
            self.show_message(
                "The plugin currently only supports the png plot output format.",
                "Please specify 'png' as the graphics output format in your parameters file.",
            )

        # Create a QThreadPool instance
        thread_pool = QThreadPool().globalInstance()

        # Create a PlotFeaturesTask instance
        task = PlotFeaturesTask(self.collection, self.params_plot)

        # connect the features_extracted signal to the handle_features_extraction_result method
        task.signals.plot_done.connect(self.handle_plot_done)

        # Connect the error signal to the handle_error method
        task.signals.error.connect(self.handle_error)

        # Start the task
        thread_pool.start(task)

        # Set the visibility of plot_indicator to True
        self.widgets["plot_indicator"].setVisible(True)

        # start the loading timer
        self.loading_timer_plot.start(2000)

    def handle_plot_done(self):
        """Handle the plot done signal."""
        # This function will be called when the PlotFeaturesTask finishes
        # stop the loading timer
        self.loading_timer_plot.stop()

        # Change the image of feature_extraction_indicator to arrow.svg
        self.widgets["plot_indicator"].setPixmap(QPixmap(self.PATH_TO_ARROW))

        self.widgets["plot_button"].setEnabled(True)

        # change the dropdown menu
        self.widgets["dropdown_loadback"].clear()

        # get all the files in the output directory, filter for self.output_path * .png
        plot_list = self.list_png_files()
        self.widgets["dropdown_loadback"].currentIndexChanged.disconnect()
        self.widgets["dropdown_loadback"].addItems(plot_list)
        self.widgets["dropdown_loadback"].currentIndexChanged.connect(
            self.on_dropdown_loadback_changed
        )

    def handle_segmentation_result(self, mask):
        """Handle the segmentation result."""
        # stop the loading timer
        self.loading_timer_segmentation.stop()

        if mask.shape != ():
            self.add_mask_to_viewer(mask)

            # Change the image of feature_extraction_indicator to arrow.svg
            self.widgets["segmentation_indicator"].setPixmap(
                QPixmap(self.PATH_TO_ARROW)
            )
        else:
            # Set the visibility of segmentation_indicator to False
            self.widgets["segmentation_indicator"].setVisible(False)

        # Re-enable the button
        self.widgets["segment_button"].setEnabled(True)

    def run_polarityjam(self):
        """Run the PolarityJam task."""
        # disable the button
        self.widgets["run_button"].setEnabled(False)

        # Create a QThreadPool instance
        thread_pool = QThreadPool().globalInstance()

        # Create a RunPolarityJamTask instance
        img, _ = self.access_image("feature")
        if img is None:
            return

        mask = self.access_mask("feature")
        if mask is None:
            return

        # override the extraction options with those from GUI
        self.params_runtime.extract_morphology_features = self.widgets[
            "extract_morphology"
        ].isChecked()
        self.params_runtime.extract_polarity_features = self.widgets[
            "extract_polarity"
        ].isChecked()
        self.params_runtime.extract_intensity_features = self.widgets[
            "extract_intensity"
        ].isChecked()
        self.params_runtime.extract_group_features = self.widgets[
            "extract_topology"
        ].isChecked()

        task = RunPolarityJamTask(
            img,
            mask,
            self.params_image,
            self.params_runtime,
            self.output_path_prefix,
            self.output_path,
        )

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
        """Handle the features extraction result."""
        # This function will be called when the RunPolarityJamTask finishes
        # The collection parameter will contain the result of the extract_features function
        self.collection = collection

        # save the collection dataset to a csv file
        self.save_feature_and_collection(collection)

        self._get_all_neighbor_combinations()

        # stop the loading timer
        self.loading_timer_feature_extraction.stop()

        # Change the image of feature_extraction_indicator to arrow.svg
        if len(collection) > 0:
            self.widgets["feature_extraction_indicator"].setPixmap(
                QPixmap(self.PATH_TO_ARROW)
            )
        else:
            # Set the visibility of feature_extraction_indicator to False
            self.widgets["feature_extraction_indicator"].setVisible(False)

        self.widgets["run_button"].setEnabled(True)

    def load_parameter_file(self):
        """Load a parameter file."""
        # Open a file dialog and load a YML file
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "", "YML Files (*.yml)"
        )

        if not file_path:
            return

        try:
            file_path = Path(file_path).resolve(strict=True)
        except FileNotFoundError:
            self.show_message("File not found!", "Please select a valid file.")

        self.params_runtime = RuntimeParameter.from_yml(file_path)
        self.params_plot = PlotParameter.from_yml(file_path)
        self.params_seg = SegmentationParameter.from_yml(file_path)

        if file_path:
            print(f"Parameter file loaded: {file_path}")

        self.widgets["param_file_loaded_indicator"].setVisible(True)

        self._param_loaded = True

    def access_image(self, call_class=None):
        """Access the image from the napari viewer."""
        image_data = None
        num_img = 0
        for layer in self.viewer.layers:
            # check if more than one image is present
            if isinstance(layer, napari.layers.image.image.Image):
                image_data = layer.data
                image_path = layer.source.path
                num_img += 1

        if num_img > 1:
            self.show_message(
                "More than one image found in the viewer!",
                "Please load only one image.",
                call_class,
            )
            return

        if image_data is None:
            self.show_message(
                "No image found in the viewer!", "Please load an image.", call_class
            )
            return

        if image_data.shape[0] < min(image_data.shape[1], image_data.shape[2]):
            img = np.swapaxes(np.swapaxes(image_data, 0, 2), 0, 1)
        else:
            img = image_data

        return img, image_path

    def access_mask(self, call_class=None):
        """Access the mask from the napari viewer."""
        mask = None
        num_mask = 0
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.labels.labels.Labels):
                mask = layer.data
                num_mask += 1

        if num_mask > 1:
            self.show_message(
                "More than one mask found in the viewer!",
                "Please have only one mask loaded.",
                call_class,
            )
            return
        if mask is None:
            self.show_message(
                "No mask found in the viewer!",
                "Please load a mask or run segmentation.",
                call_class,
            )
            return

        return mask

    def add_mask_to_viewer(self, mask, name="PolarityJam Mask"):
        """Add the mask to the napari viewer."""
        self.viewer.add_labels(mask, name=name)
