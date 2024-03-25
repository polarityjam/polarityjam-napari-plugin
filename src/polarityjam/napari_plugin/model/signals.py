import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from polarityjam import PropertiesCollection


class WorkerSignalsPlot(QObject):
    plot_done = pyqtSignal()  # Signal that will be emitted when the plot is done
    error = pyqtSignal(tuple)


class WorkerSignalsExtraction(QObject):
    features_extracted = pyqtSignal(PropertiesCollection)  # Signal that will be emitted when features are extracted
    error = pyqtSignal(tuple)


class WorkerSignalsSegmentation(QObject):
    finished = pyqtSignal(np.ndarray)  # Signal that will be emitted when the task finishes
    error = pyqtSignal(tuple)
