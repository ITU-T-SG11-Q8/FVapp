from PyQt5 import uic
from PyQt5.QtWidgets import QDialog


class RoomInformationClass(QDialog):
    def __init__(self, p_modify_information_room):
        super().__init__()
        self.ui = uic.loadUi("GUI/ROOM_INFORMATION.ui", self)
        self.button_ok.clicked.connect(p_modify_information_room)
        self.button_cancel.clicked.connect(self.close_information_room)

    def close_information_room(self):
        self.close()

