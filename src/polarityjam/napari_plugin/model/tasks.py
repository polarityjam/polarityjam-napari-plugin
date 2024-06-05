"""QRunnable Tasks for running PolarityJam-napari in a separate thread."""
import traceback

import numpy as np
from PyQt5.QtCore import QRunnable, pyqtSlot

from polarityjam import (
    Extractor,
    Plotter,
    PropertiesCollection,
    SegmentationParameter,
    load_segmenter,
)
from polarityjam.napari_plugin.model.signals import (
    WorkerSignalsExtraction,
    WorkerSignalsPlot,
    WorkerSignalsSegmentation,
)


class PlotFeaturesTask(QRunnable):
    """Task for plotting features."""

    def __init__(self, collection, params_plot):
        """Initialize the task."""
        super().__init__()
        self.signals = WorkerSignalsPlot()
        self.collection = collection
        self.params_plot = params_plot

    @pyqtSlot()
    def run(self):
        """Run the task."""
        try:
            self.plot_features()
        except Exception as e:
            self.signals.error.emit((e, "Plot"))
            traceback.print_exc()

    def plot_features(self):
        """Plot the features."""
        plotter = Plotter(self.params_plot)
        plotter.plot_collection(self.collection)

        self.signals.plot_done.emit()


class RunPolarityJamTask(QRunnable):
    """Task for running PolarityJam."""

    def __init__(
        self, img, mask, params_image, params_runtime, output_path_prefix, output_path
    ):
        """Initialize the task."""
        super().__init__()
        self.signals = WorkerSignalsExtraction()
        self.img = img
        self.mask = mask
        self.params_image = params_image
        self.params_runtime = params_runtime
        self.output_path_prefix = output_path_prefix
        self.output_path = output_path

    @pyqtSlot()
    def run(self):
        """Run the task."""
        try:
            collection = self.extract_features()
        except Exception as e:
            self.signals.error.emit((e, "Extraction"))
            traceback.print_exc()
            collection = PropertiesCollection()
        self.signals.features_extracted.emit(collection)

    def extract_features(self):
        """Extract features."""
        collection = PropertiesCollection()
        extractor = Extractor(self.params_runtime)

        extractor.extract(
            self.img,
            self.params_image,
            self.mask,
            self.output_path_prefix,
            self.output_path,
            collection,
        )

        return collection


class RunSegmentationTask(QRunnable):
    """Task for running segmentation."""

    def __init__(self, img, params_seg, params_runtime, params_image):
        """Initialize the task."""
        super().__init__()
        self.img = img
        self.params_seg = params_seg
        self.params_runtime = params_runtime
        self.params_image = params_image
        self.signals = WorkerSignalsSegmentation()

    @pyqtSlot()
    def run(self):
        """Run the task."""
        try:
            mask = self.segment_image()
        except Exception as e:
            self.signals.error.emit((e, "Segmentation"))
            traceback.print_exc()
            mask = np.array(np.nan)

        self.signals.finished.emit(mask)

    def segment_image(self):
        """Segment the image."""
        if self.params_seg is None:
            self.params_seg = SegmentationParameter(
                self.params_runtime.segmentation_algorithm
            )

        segmenter, _ = load_segmenter(self.params_runtime, self.params_seg)

        img_channels, _ = segmenter.prepare(self.img, self.params_image)
        mask = segmenter.segment(img_channels)

        return mask
