import numpy as np
from PyQt5.QtCore import QRunnable, pyqtSlot
from polarityjam import Plotter, PropertiesCollection, Extractor, SegmentationParameter, load_segmenter

from polarityjam.napari_plugin.model.signals import WorkerSignalsPlot, WorkerSignalsExtraction, WorkerSignalsSegmentation


class PlotFeaturesTask(QRunnable):
    def __init__(self, collection, params_plot):
        super().__init__()
        self.signals = WorkerSignalsPlot()
        self.collection = collection
        self.params_plot = params_plot

    @pyqtSlot()
    def run(self):
        try:
            self.plot_features()
        except Exception as e:
            self.signals.error.emit((e, "Plot"))

    def plot_features(self):
        plotter = Plotter(self.params_plot)
        plotter.plot_collection(self.collection)

        self.signals.plot_done.emit()


class RunPolarityJamTask(QRunnable):
    def __init__(self, img, mask, params_image, params_runtime, output_path_prefix, output_path):
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
        try:
            collection = self.extract_features()
        except Exception as e:
            self.signals.error.emit((e, "Extraction"))
            collection = PropertiesCollection()
        self.signals.features_extracted.emit(collection)

    def extract_features(self):
        collection = PropertiesCollection()
        extractor = Extractor(self.params_runtime)

        extractor.extract(self.img, self.params_image, self.mask, self.output_path_prefix, self.output_path, collection)

        return collection


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
            self.signals.error.emit((e, "Segmentation"))

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
            mask = np.NAN

        return mask
