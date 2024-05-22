from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QDialog
from PyQt5 import QtCore


class RenderViewClass(QDialog):
    update_stat_signal = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("GUI/VIDEO_RENDERER.ui", self)

        self.update_stat_signal.connect(self.update_stat)

    @pyqtSlot(str)
    def update_stat(self, stat):
        self.label_stat.setText(stat)

    def request_update_stat(self, stat):
        self.update_stat_signal.emit(stat)
