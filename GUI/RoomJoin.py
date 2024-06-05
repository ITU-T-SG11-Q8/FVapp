from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QFileDialog
import hp2papi as api
from typing import List


class RoomOverlay:
    def __init__(self, overlayId: str, title: str, ownerId: str):
        self.overlayId: str = overlayId
        self.title: str = title
        self.ownerId: str = ownerId


class RoomJoinClass(QDialog):
    def __init__(self, p_send_join_room_func):
        super().__init__()
        self.ui = uic.loadUi("GUI/JOIN_ROOM.ui", self)
        self.comboBox_overlay_id.currentIndexChanged.connect(self.on_changed_overlay_id)
        self.button_ok.clicked.connect(p_send_join_room_func)
        self.button_cancel.clicked.connect(self.on_close_button)
        self.button_query.clicked.connect(self.on_overlay_id_search_func)
        self.button_search_private.clicked.connect(self.on_search_private)
        self.overlays: List[RoomOverlay] = []
        self.owner_id = ''

    def clear_value(self):
        self.comboBox_overlay_id.clear()
        self.overlays.clear()
        self.lineEditTitle.setText('')
        self.lineEdit_peer_id.setText('')
        self.lineEdit_display_name.setText('')
        self.lineEdit_private_key.setText('')

    def on_search_private(self):
        private_key = QFileDialog.getOpenFileName(self, filter='*.pem')
        self.lineEdit_private_key.setText(private_key[0])

    def on_close_button(self):
        self.close()

    def on_changed_overlay_id(self, index):
        if 0 <= index < len(self.overlays):
            self.owner_id = self.overlays[index].ownerId
            self.lineEditTitle.setText(self.overlays[index].title)

    def on_overlay_id_search_func(self):
        self.owner_id = ''
        self.comboBox_overlay_id.clear()
        self.overlays.clear()

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
        if len(query_res.overlay) > 0:
            self.owner_id = query_res.overlay[0].ownerId

            for i in query_res.overlay:
                print(f'add overlay:{i.overlayId} title:{i.title}')

                room_overlay: RoomOverlay = RoomOverlay(i.overlayId, i.title, i.ownerId)
                self.overlays.append(room_overlay)

                self.ui.comboBox_overlay_id.addItem(i.overlayId)

