from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QFileDialog
import hp2papi as api
from typing import List


class RoomOverlay:
    def __init__(self, overlayId: str, title: str):
        self.overlayId: str = overlayId
        self.title: str = title


class RoomJoinClass(QDialog):
    def __init__(self, p_send_join_room_func):
        super().__init__()
        self.ui = uic.loadUi("GUI/JOIN_ROOM.ui", self)
        self.comboBox_overlay_id.currentIndexChanged.connect(self.on_comboBox_overlay_id_changed)
        self.button_ok.clicked.connect(p_send_join_room_func)
        self.button_cancel.clicked.connect(self.close_button)
        self.button_query.clicked.connect(self.overlay_id_search_func)
        self.button_search_private.clicked.connect(self.search_private)
        self.overlays: List[RoomOverlay] = []

    def clear_value(self):
        self.comboBox_overlay_id.clear()
        self.overlays.clear()
        self.lineEditTitle.setText('')
        self.lineEdit_peer_id.setText('')
        self.lineEdit_display_name.setText('')
        self.lineEdit_private_key.setText('')

    def search_private(self):
        private_key = QFileDialog.getOpenFileName(self, filter='*.pem')
        self.lineEdit_private_key.setText(private_key[0])

    def close_button(self):
        self.close()

    def on_comboBox_overlay_id_changed(self, index):
        if index < len(self.overlays):
            title = self.overlays[index].title
            self.lineEditTitle.setText(title)

    def overlay_id_search_func(self):
        self.comboBox_overlay_id.clear()
        self.overlays.clear()
        self.lineEditTitle.setText('')

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
            print(f'add overlay:{i.overlayId} title:{i.title}')

            roomOverlay: RoomOverlay = RoomOverlay(i.overlayId, i.title)
            self.overlays.append(roomOverlay)

            self.ui.comboBox_overlay_id.addItem(i.overlayId)
