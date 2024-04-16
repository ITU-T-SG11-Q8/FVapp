from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QFileDialog
import hp2papi as api


class RoomJoinClass(QDialog):
    def __init__(self, p_send_join_room_func):
        super().__init__()
        self.ui = uic.loadUi("GUI/JOIN_ROOM.ui", self)
        self.button_ok.clicked.connect(p_send_join_room_func)
        self.button_cancel.clicked.connect(self.close_button)
        self.button_query.clicked.connect(self.overlay_id_search_func)
        self.button_search_private.clicked.connect(self.search_private)

    def search_private(self):
        private_key = QFileDialog.getOpenFileName(self)
        self.lineEdit_private_key.setText(private_key[0])

    def close_button(self):
        self.close()

    def overlay_id_search_func(self):
        query_res = api.Query()
        if query_res.code is not api.ResponseCode.Success:
            print("\nQuery fail.")
            return
        else:
            print("\nQuery success.")

        print("\nOverlays:", query_res.overlay)

        if len(query_res.overlay) <= 0:
            print("\noverlay id empty.")

        # query_len = len(query_res.overlay)
        for i in query_res.overlay:
            print(f'add overlay:{i.overlayId} ')
            self.ui.comboBox_overlay_id.addItem(i.overlayId)
