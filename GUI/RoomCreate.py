from PyQt5 import uic
from PyQt5.QtWidgets import QDialog


class RoomCreateClass(QDialog):
    def __init__(self, p_create_room_ok_func):
        super().__init__()
        self.ui = uic.loadUi("GUI/ROOM_CREATE.ui", self)
        self.button_ok.clicked.connect(p_create_room_ok_func)
        self.button_cancel.clicked.connect(self.close_button)

    def close_button(self):
        self.close()

    def clear_value(self):
        self.lineEdit_title.setText("")
        self.lineEdit_description.setText("")
        self.lineEdit_ower_id.setText("")
        self.lineEdit_admin_key.setText("")
        self.checkBox_facevideo.setChecked(True)
        self.checkBox_audio.setChecked(True)
        self.checkBox_text.setChecked(True)

