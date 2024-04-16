from PyQt5.QtWidgets import QDialog
from PyQt5 import QtGui, uic


class RenderViewClass(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("GUI/VIDEO_RENDERER.ui", self)
