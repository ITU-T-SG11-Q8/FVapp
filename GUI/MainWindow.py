import os

import pyaudio
from PyQt5 import QtCore, uic, QtGui
from PyQt5.QtCore import QTimer, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox

from GUI.RoomCreate import RoomCreateClass
from GUI.RoomInformation import RoomInformationClass
from GUI.RoomJoin import RoomJoinClass
from afy.arguments import opt
from gooroomee.grm_defs import ModeType, SessionData, PeerData, IMAGE_SIZE, GrmParentThread, MediaQueueData, OwnerData
from gooroomee.grm_packet import BINWrapper, TYPE_INDEX
from typing import List
from afy.utils import resize

import hp2papi as api
import time
import pygame
import pygame.camera
import cv2
import numpy as np

form_class = uic.loadUiType("GUI/MAIN_WINDOW.ui")[0]
owner_data: OwnerData = OwnerData()


def internal_create_channel_audio():
    audio_channel = api.ChannelAudio()
    audio_channel.channelId = "audioChannel"
    audio_channel.codec = api.AudioCodec.AAC
    audio_channel.sampleRate = api.AudioSampleRate.Is44100
    audio_channel.bitrate = api.AudioBitrate.Is128kbps
    audio_channel.audioChannelType = api.AudioChannelType.Stereo
    audio_channel.sourceList = ["*"]
    return audio_channel


def internal_create_channel_text():
    text_channel = api.ChannelText()
    text_channel.channelId = "textChannel"
    text_channel.format = api.TextFormat.Plain
    text_channel.encoding = api.TextEncoding.UTF8
    text_channel.sourceList = ["*"]
    return text_channel


def internal_create_channel_facevideo(mode_type):
    face_video_channel = api.ChannelFeatureBasedVideo()
    # channelId는 channel의 key로 사용. 임의의 값을 사용할 수 있음
    # 추후 channel의 속성 변경을 위해 알고 있어야 함
    face_video_channel.target = api.FeatureBasedVideoTarget.Face
    if mode_type == ModeType.KDM:
        face_video_channel.channelId = "kdmChannel"
        face_video_channel.mode = api.FeatureBasedVideoMode.KeypointsDescriptionMode
        face_video_channel.resolution = f"{IMAGE_SIZE}x{IMAGE_SIZE}"
        face_video_channel.framerate = "30fps"
        face_video_channel.keypointsType = "68points"
    else:
        face_video_channel.channelId = "snnmChannel"
        face_video_channel.mode = api.FeatureBasedVideoMode.SharedNeuralNetworkMode
        face_video_channel.modelUri = "modeUrl"
        face_video_channel.hash = "hash"
        face_video_channel.dimension = "dimension"
    # channel의 sourcelist를 설정하면 해당 사용자만 해당 채널의 데이터 송신 가능
    # 설정하지 않으면 CreationRequest의 sourceList를 따름
    # 설정하면 해당 channel에서는 CreationRequest의 sourceList는 무시됨
    # 값이 없는 리스트 []를 설정하면 해당 채널의 데이터 송신 불가능
    # Owner 권한이 있는 사용자가 Modification을 통해 sourceList를 변경할 수 있음
    face_video_channel.sourceList = ["*"]
    return face_video_channel


def get_current_time_ms():
    return round(time.time() * 1000)


class MainWindowClass(QMainWindow, form_class):
    mode_type: ModeType = ModeType.SNNM
    reference_image_frame = None
    terminated_room_signal = QtCore.pyqtSignal()
    update_log_signal = QtCore.pyqtSignal(str)
    update_stat_signal = QtCore.pyqtSignal(str)
    alert_signal = QtCore.pyqtSignal(str, str)
    log_str: str = ""
    log_index: int = 0

    def __init__(self,
                 p_get_worker_seq_num,
                 p_get_worker_ssrc,
                 p_set_join):
        super().__init__()
        self.setupUi(self)
        self.join_session: SessionData = SessionData()
        self.join_peer: List[PeerData] = []

        self.camera_device_init()
        self.audio_device_init()

        self.keyframe_period = opt.keyframe_period
        if self.keyframe_period is None:
            self.keyframe_period = 0

        print(f'key frame period:{self.keyframe_period}')

        self.send_chat_queue = None  # GRMQueue
        self.worker_capture_frame = None  # CaptureFrameWorker
        self.worker_video_encode_packet = None  # EncodeVideoPacketWorker
        self.worker_video_decode_and_render_packet = None  # DecodeAndRenderVideoPacketWorker
        self.worker_mic_encode_packet = None  # EncodeMicPacketWorker
        self.worker_speaker_decode_packet = None  # DecodeSpeakerPacketWorker
        self.worker_grm_comm = None  # GrmCommWorker

        self.get_worker_seq_num = p_get_worker_seq_num
        self.get_worker_ssrc = p_get_worker_ssrc
        self.set_join = p_set_join

        self.create_button.clicked.connect(self.on_create_room)
        self.join_button.clicked.connect(self.on_join_room)
        self.room_information_button.clicked.connect(self.on_information_room)
        self.button_chat_send.clicked.connect(self.on_send_chat)
        self.lineEdit_input_chat.returnPressed.connect(self.on_send_chat)
        self.comboBox_mic_device.currentIndexChanged.connect(self.on_change_mic_device)
        self.comboBox_spk_device.currentIndexChanged.connect(self.on_change_spk_device)
        self.comboBox_camera_device.currentIndexChanged.connect(self.on_change_camera_device)
        self.button_exit.clicked.connect(self.on_exit_button)
        self.checkBox_use_reference_image.stateChanged.connect(self.on_change_use_reference_image)
        self.button_search_reference_image.clicked.connect(self.on_search_reference_image)
        self.listWidget.itemSelectionChanged.connect(self.on_user_selection_changed)

        self.button_chat_send.setDisabled(True)
        self.lineEdit_input_chat.setDisabled(True)
        self.overlay_id = ""
        self.peer_id = ""
        self.title = ""
        self.display_name = ""
        self.private_key = ""
        self.timer = QTimer(self)

        if self.keyframe_period > 0:
            self.timer.start(self.keyframe_period)
            self.timer.timeout.connect(self.timeout_key_frame)

        self.bin_wrapper = BINWrapper()

        self.room_create_ui = RoomCreateClass(self.create_room_ok_func)
        self.join_ui = RoomJoinClass(self.send_join_room_func)
        self.room_information_ui = RoomInformationClass(self.modify_information_room)

        self.terminated_room_signal.connect(self.terminated_room)
        self.update_log_signal.connect(self.update_log)
        self.update_stat_signal.connect(self.update_stat)
        self.alert_signal.connect(self.alert)

    def set_workers(self,
                    p_send_chat_queue,
                    p_worker_capture_frame,
                    p_worker_video_encode_packet,
                    p_worker_video_decode_and_render_packet,
                    p_worker_mic_encode_packet,
                    p_worker_speaker_decode_packet,
                    p_worker_grm_comm,
                    p_reference_image):
        self.send_chat_queue = p_send_chat_queue  # GRMQueue
        self.worker_capture_frame = p_worker_capture_frame  # CaptureFrameWorker
        self.worker_video_encode_packet = p_worker_video_encode_packet  # EncodeVideoPacketWorker
        self.worker_video_decode_and_render_packet = p_worker_video_decode_and_render_packet  # DecodeAndRenderVideoPacketWorker
        self.worker_mic_encode_packet = p_worker_mic_encode_packet  # EncodeMicPacketWorker
        self.worker_speaker_decode_packet = p_worker_speaker_decode_packet  # DecodeSpeakerPacketWorker
        self.worker_grm_comm = p_worker_grm_comm  # GrmCommWorker

        if self.worker_video_encode_packet is not None:
            self.button_send_keyframe.clicked.connect(self.worker_video_encode_packet.request_send_key_frame)
            self.button_request_keyframe.clicked.connect(self.worker_video_encode_packet.request_recv_key_frame)

            if p_reference_image is not None:
                self.checkBox_use_reference_image.setChecked(True)

                if self.apply_reference_image(p_reference_image) is False:
                    self.checkBox_use_reference_image.setChecked(False)

    @pyqtSlot()
    def terminated_room(self):
        self.leave_room()
        self.remove_room(False)

    def request_terminated_room(self):
        self.terminated_room_signal.emit()

    @pyqtSlot(str)
    def update_log(self, log):
        self.label_log.setText(log)

    def request_update_log(self, log):
        self.log_str = '[%3d] %s' % (self.log_index, log) + '\n' + self.log_str
        self.log_index += 1

        self.update_log_signal.emit(self.log_str)

    @pyqtSlot(str)
    def update_stat(self, log):
        self.label_stat.setText(log)

    def request_update_stat(self, camera_fps):
        video_encode_queue_count = 0
        video_send_video_fps = 0
        video_send_queue_count = 0
        video_recv_queue_count = 0

        if self.worker_video_encode_packet and self.worker_video_encode_packet.video_capture_queue:
            video_encode_queue_count = self.worker_video_encode_packet.video_capture_queue.length()
        if self.worker_grm_comm and self.worker_grm_comm.send_video_queue:
            video_send_video_fps = self.worker_grm_comm.get_send_video_fps()
            video_send_queue_count = self.worker_grm_comm.send_video_queue.length()
        if self.worker_grm_comm and self.worker_grm_comm.recv_video_queue:
            video_recv_queue_count = self.worker_grm_comm.recv_video_queue.length()

        stat = f'camera capture fps : {camera_fps}\n' \
               f'video encode queue count : {video_encode_queue_count}\n' \
               f'video send fps : {video_send_video_fps}\n' \
               f'video send queue count : {video_send_queue_count}\n' \
               f'video recv queue count : {video_recv_queue_count}'

        self.update_stat_signal.emit(stat)

    def request_alert(self, title, message):
        self.alert_signal.emit(title, message)

    @pyqtSlot(str, str)
    def alert(self, title, message):
        self.update_log(f'[{title}] {message}')
        QMessageBox.information(self, title, message)

    def timeout_key_frame(self):
        if self.worker_video_encode_packet is not None:
            self.worker_video_encode_packet.request_send_key_frame()

    def camera_device_init(self):
        pygame.camera.init()
        # make the list of all available cameras
        cam_list = pygame.camera.list_cameras()
        print(f"cam_list:{cam_list}")
        for index in cam_list:
            device_string = "Camera #" + str(index)
            self.comboBox_camera_device.addItem(device_string, userData=index)

    def audio_device_init(self):
        pa = pyaudio.PyAudio()
        for device_index in range(pa.get_device_count()):
            audio_info = pa.get_device_info_by_index(device_index)
            device_name = ""
            index = ""
            for key in audio_info.keys():
                # print(key, ' = ', info[key])
                if key == "index":
                    index = audio_info[key]
                if key == "name":
                    device_name = audio_info[key]
                if key == "maxInputChannels":
                    if audio_info[key] == 0:
                        print(f"Input deviceName:{device_name}, index:{index}")
                        self.comboBox_spk_device.addItem(device_name, userData=index)
                if key == "maxOutputChannels":
                    if audio_info[key] == 0:
                        print(f"Output deviceName:{device_name}, index:{index}")
                        self.comboBox_mic_device.addItem(device_name, userData=index)

    def on_create_room(self):
        if self.create_button.text() == "Channel Create":
            self.room_create_ui.clear_value()

            self.room_create_ui.lineEdit_ower_id.setText(owner_data.ownerPeerId)
            self.room_create_ui.lineEdit_admin_key.setText(owner_data.ownerAdminKey)
            self.room_create_ui.checkBox_facevideo.setChecked(owner_data.ownerUseFaceVideo)
            self.room_create_ui.checkBox_audio.setChecked(owner_data.ownerUseAudio)
            self.room_create_ui.checkBox_text.setChecked(owner_data.ownerUseText)

            self.room_create_ui.radioButton_snnm.setChecked(False)
            self.room_create_ui.radioButton_kdm.setChecked(False)
            if owner_data.ownerModeType == ModeType.KDM:
                self.room_create_ui.radioButton_kdm.setChecked(True)
            else:
                self.room_create_ui.radioButton_snnm.setChecked(True)

            self.room_create_ui.show()

        elif self.create_button.text() == "Channel Delete":
            self.remove_room(True)

    def create_room_ok_func(self):
        if self.create_button.text() == "Channel Create":
            title = self.room_create_ui.lineEdit_title.text()
            description = self.room_create_ui.lineEdit_description.text()
            owner_id = self.room_create_ui.lineEdit_ower_id.text()
            admin_key = self.room_create_ui.lineEdit_admin_key.text()
            checked_face_video = self.room_create_ui.checkBox_facevideo.isChecked()
            checked_audio = self.room_create_ui.checkBox_audio.isChecked()
            checked_text = self.room_create_ui.checkBox_text.isChecked()

            if len(title) == 0:
                self.request_alert('alert', 'Please enter the title.')
                return
            if len(description) == 0:
                self.request_alert('alert', 'Please enter the description.')
                return
            if len(owner_id) == 0:
                self.request_alert('alert', 'Please enter the owner_id.')
                return
            if len(admin_key) == 0:
                self.request_alert('alert', 'Please enter the admin_key.')
                return

            owner_data.set_value_peer_id(owner_id)
            owner_data.set_value_admin_key(admin_key)
            owner_data.set_value_use_face_video(checked_face_video)
            owner_data.set_value_use_audio(checked_audio)
            owner_data.set_value_use_text(checked_text)
            if self.room_create_ui.radioButton_kdm.isChecked() is True:
                owner_data.set_value_mode_type(ModeType.KDM)
            else:
                owner_data.set_value_mode_type(ModeType.SNNM)
            owner_data.write_values()

            if self.room_create_ui.radioButton_kdm.isChecked() is True:
                self.mode_type = ModeType.KDM
                self.request_update_log(f'Request to create KDM mode channel. title:{title}')
            else:
                self.mode_type = ModeType.SNNM
                self.request_update_log(f'Request to create SNNM mode channel. title:{title}')

            creation_req = api.CreationRequest(title=title, description=description,
                                               ownerId=owner_id, adminKey=admin_key)

            service_control_channel = api.ChannelServiceControl()
            service_control_channel.channelId = "controlChannel"

            channel_list = []
            if checked_face_video is True:
                face_video_channel = internal_create_channel_facevideo(self.mode_type)
                channel_list.append(face_video_channel)
            if checked_audio is True:
                audio_channel = internal_create_channel_audio()
                channel_list.append(audio_channel)
            if checked_text is True:
                text_channel = internal_create_channel_text()
                channel_list.append(text_channel)
            creation_req.channelList = channel_list

            print("\nCreationRequest:", creation_req)
            creation_req.sourceList = ["*"]
            creation_res = api.Creation(creation_req)
            self.room_create_ui.close()

            print("\nCreationResponse:", creation_res)

            if creation_res.code is api.ResponseCode.Success:
                print("\nCreation success.", creation_res.overlayId)
                if self.room_create_ui.radioButton_kdm.isChecked() is True:
                    self.request_update_log(f'Succeed to create KDM mode channel. overlayId:{creation_res.overlayId}')
                else:
                    self.request_update_log(f'Succeed to create SNNM mode channel. overlayId:{creation_res.overlayId}')

                self.join_session.creationOverlayId = creation_res.overlayId
                self.join_session.creationTitle = title

                self.join_session.overlayId = creation_res.overlayId
                self.join_session.ownerId = owner_id
                self.join_session.adminKey = admin_key

                self.create_button.setText("Channel Delete")
                self.room_information_button.setDisabled(False)

                print('register session_notification_listener by creation')
                api.SetNotificatonListener(self.join_session.overlayId, self.join_session.ownerId,
                                           func=self.session_notification_listener)
            else:
                self.request_alert('error', f'Failed to create the channel. error code:{creation_res.code}')
                self.join_session.overlayId = ""

        elif self.create_button.text() == "Channel Delete":
            self.remove_room(True)
            self.all_stop_worker()
            self.room_create_ui.close()

    def on_join_room(self):
        if self.join_button.text() == "Channel Join":
            self.join_ui.clear_value()

            if self.join_session.creationOverlayId is not None and len(self.join_session.creationOverlayId) > 0:
                self.join_ui.comboBox_overlay_id.addItem(self.join_session.creationOverlayId)
                self.join_ui.lineEditTitle.setText(self.join_session.creationTitle)

            self.join_ui.lineEdit_peer_id.setText(owner_data.ownerPeerId)
            self.join_ui.lineEdit_display_name.setText(owner_data.ownerDisplayName)
            self.join_ui.lineEdit_private_key.setText(owner_data.ownerPrivateKey)

            self.join_ui.show()
        elif self.join_button.text() == "Channel Leave":
            self.leave_room()

    def send_join_room_func(self):
        if self.join_button.text() == "Channel Join":
            self.overlay_id = self.join_ui.comboBox_overlay_id.currentText()
            self.peer_id = self.join_ui.lineEdit_peer_id.text()
            self.title = self.join_ui.lineEditTitle.text()
            self.display_name = self.join_ui.lineEdit_display_name.text()
            self.private_key = self.join_ui.lineEdit_private_key.text()

            if len(self.overlay_id) == 0:
                self.request_alert('alert', 'Please enter the overlay_id.')
                return
            if len(self.peer_id) == 0:
                self.request_alert('alert', 'Please enter the peer_id.')
                return
            if len(self.display_name) == 0:
                self.request_alert('alert', 'Please enter the display_name.')
                return
            if len(self.private_key) == 0:
                self.request_alert('alert', 'Please enter the private_key.')
                return

            owner_data.set_value_peer_id(self.peer_id)
            owner_data.set_value_display_name(self.display_name)
            owner_data.set_value_private_key(self.private_key)
            owner_data.write_values()

            self.room_information_button.setDisabled(False)
            self.create_button.setDisabled(True)

            self.button_chat_send.setDisabled(False)
            self.lineEdit_input_chat.setDisabled(False)

            self.join_session.overlayId = self.overlay_id
            self.join_session.title = self.title
            self.join_session.ownerId = self.peer_id
            self.room_information_button.setDisabled(False)

            print('register session_notification_listener by join')
            api.SetNotificatonListener(overlayId=self.join_session.overlayId,
                                       peerId=self.join_session.ownerId,
                                       func=self.session_notification_listener)

            self.request_update_log(f'Request to join the channel')
            private_key_abs = os.path.abspath(self.private_key)
            join_request = api.JoinRequest(self.overlay_id, "", self.peer_id, self.display_name, private_key_abs)
            print("\nJoinRequest:", join_request)
            join_response = api.Join(join_request)
            print("\nJoinResponse:", join_response)
            self.join_ui.close()

            if join_response.code is api.ResponseCode.Success:
                if self.worker_capture_frame is not None:
                    if self.mode_type == ModeType.KDM:
                        self.worker_capture_frame.set_capture_fps(10)
                    else:
                        self.worker_capture_frame.set_capture_fps(20)

                # if the owner is me, the channel must be deletable
                self.join_button.setText("Channel Leave")
                self.join_session.channelList = join_response.channelList
                self.mode_type = self.join_session.video_channel_type()
                self.join_session.title = join_response.title
                self.join_session.description = join_response.description
                self.set_join(True)

                if len(self.join_ui.owner_id) > 0 and self.join_ui.owner_id == owner_data.ownerPeerId:
                    self.join_session.creationOverlayId = self.join_session.overlayId
                    self.join_session.creationTitle = self.join_session.title
                    self.create_button.setText("Channel Delete")

                self.request_update_log(f'Succeed to join the channel')

                if self.join_session.channelList is not None:
                    for i in self.join_session.channelList:
                        if i.channelType is api.ChannelType.FeatureBasedVideo:
                            if self.worker_capture_frame is not None:
                                self.worker_capture_frame.set_enable_cam(True)
                        elif i.channelType is api.ChannelType.Audio:
                            if self.worker_mic_encode_packet is not None:
                                self.worker_mic_encode_packet.set_enable_mic(True)
                            if self.worker_speaker_decode_packet is not None:
                                self.worker_speaker_decode_packet.set_enable_spk(True)
            else:
                self.request_alert('error', f'Failed to join the channel. error code:{join_response.code}')

            return join_response

        elif self.join_button.text() == "Channel Leave":
            self.leave_room()

    def leave_room(self):
        self.request_update_log(f'Request to leave the channel.')

        self.set_join(False)
        self.join_button.setText("Channel Join")
        self.create_button.setDisabled(False)

        while self.worker_grm_comm.is_stopped_comm() is False:
            time.sleep(1)

        res = api.Leave(api.LeaveRequest(overlayId=self.join_session.overlayId,
                                         peerId=self.peer_id,
                                         accessKey=self.join_session.accessKey))

        if res.code is api.ResponseCode.Success:
            self.request_update_log(f'Succeed to leave the channel.')
        else:
            self.request_alert('error', f'Failed to leave the channel. error code:{res.code}')
            print("\nLeave fail.", res.code)
            
        self.join_peer.clear()
        self.join_session.clear_value()
        self.listWidget.clear()
        self.listWidget_chat_message.clear()

    def on_information_room(self):
        if self.join_session.overlayId is not None:
            self.room_information_ui.lineEdit_overlay_id.setText(self.join_session.overlayId)

        if self.join_session.ownerId is not None:
            self.room_information_ui.lineEdit_ower_id.setText(self.join_session.ownerId)

        if self.join_session.adminKey is not None:
            self.room_information_ui.lineEdit_admin_key.setText(self.join_session.adminKey)

        if self.join_session.title is not None:
            self.room_information_ui.lineEdit_title.setText(self.join_session.title)

        if self.join_session.description is not None:
            self.room_information_ui.lineEdit_description.setText(self.join_session.description)

        if self.join_session.adminKey is not None and len(self.join_session.adminKey) > 0:
            self.room_information_ui.button_ok.setDisabled(False)
        else:
            self.room_information_ui.button_ok.setDisabled(True)

        if self.join_session.video_channel_type() == ModeType.KDM:
            self.room_information_ui.radioButton_snnm.setChecked(False)
            self.room_information_ui.radioButton_kdm.setChecked(True)
        else:
            self.room_information_ui.radioButton_snnm.setChecked(True)
            self.room_information_ui.radioButton_kdm.setChecked(False)

        self.room_information_ui.groupBox.setCheckable(False)
        self.room_information_ui.checkBox_facevideo.setChecked(False)
        self.room_information_ui.checkBox_audio.setChecked(False)
        self.room_information_ui.checkBox_text.setChecked(False)

        if self.join_session.channelList is not None:
            for i in self.join_session.channelList:
                if i.channelType is api.ChannelType.FeatureBasedVideo:
                    self.room_information_ui.checkBox_facevideo.setChecked(True)
                elif i.channelType is api.ChannelType.Audio:
                    self.room_information_ui.checkBox_audio.setChecked(True)
                elif i.channelType is api.ChannelType.Text:
                    self.room_information_ui.checkBox_text.setChecked(True)
        self.room_information_ui.show()

    def modify_information_room(self):
        self.request_update_log(f'Request to modify the information channel')
        print("Modify Information Room")
        title = self.room_information_ui.lineEdit_title.text()
        description = self.room_information_ui.lineEdit_description.text()

        modification_req = api.ModificationRequest(overlayId=self.join_session.overlayId,
                                                   ownerId=self.join_session.ownerId,
                                                   adminKey=self.join_session.accessKey)

        # 변경할 값만 입력
        modification_req.adminKey = self.join_session.adminKey
        modification_req.title = title
        modification_req.description = description

        channel_list = []
        if self.room_information_ui.checkBox_facevideo.isChecked():
            face_video_channel = internal_create_channel_facevideo(self.mode_type)
            face_video_channel.sourceList = ["*"]
            channel_list.append(face_video_channel)
        if self.room_information_ui.checkBox_audio.isChecked():
            audio_channel = internal_create_channel_audio()
            audio_channel.sourceList = ["*"]
            channel_list.append(audio_channel)
        if self.room_information_ui.checkBox_text.isChecked():
            text_channel = internal_create_channel_text()
            text_channel.sourceList = ["*"]
            channel_list.append(text_channel)
        modification_req.channelList = channel_list

        # modification_req.newOwnerId = self.join_session.ownerId
        # modification_req.newAdminKey = self.join_session.accessKey
        # modification_req.startDateTime = "20230101090000"
        # modification_req.endDateTime = "20230101100000"
        # modification_req.accessKey = "new_access_key"
        # modification_req.peerList = ["user3", "user4"]
        # modification_req.blockList = ["user5"]

        # modification_req.sourceList = ["*"]

        # video_channel = api.ChannelFeatureBasedVideo()
        # video_channel.sourceList = ["*"]
        # modification_req.channelList = [video_channel]

        print("\nModificationRequest:", modification_req)

        modification_res = api.Modification(modification_req)

        print("\nModificationResponse:", modification_res)

        if modification_res.code is api.ResponseCode.Success:
            self.request_update_log(f'Succeed to modify the information channel')
            print("\nModification success.")
        else:
            self.request_alert('error', f'Failed to modify the channel information. error code:{modification_res.code}')

        self.room_information_ui.close()

    def on_send_chat(self):
        print('send chat')
        input_message = self.lineEdit_input_chat.text()
        self.output_chat(self.display_name, None, input_message)
        self.lineEdit_input_chat.clear()

        chat_message = self.bin_wrapper.to_bin_chat_data(input_message)
        chat_message = self.bin_wrapper.to_bin_wrap_common_header(timestamp=get_current_time_ms(),
                                                                  seq_num=self.get_worker_seq_num(),
                                                                  ssrc=self.get_worker_ssrc(),
                                                                  mediatype=TYPE_INDEX.TYPE_DATA,
                                                                  bindata=chat_message)
        self.send_chat_queue.put(chat_message)

    def on_change_mic_device(self):
        if self.worker_mic_encode_packet is not None:
            self.worker_mic_encode_packet.change_device_mic(self.comboBox_mic_device.currentData())

    def on_change_spk_device(self):
        if self.worker_speaker_decode_packet is not None:
            self.worker_speaker_decode_packet.change_device_spk(self.comboBox_spk_device.currentData())

    def on_change_camera_device(self):
        if self.worker_capture_frame is not None:
            self.worker_capture_frame.change_device_cam(self.comboBox_camera_device.currentData())

    def on_exit_button(self):
        threads: List[GrmParentThread] = [
            self.worker_video_encode_packet,
            self.worker_video_decode_and_render_packet,
            self.worker_capture_frame,
            self.worker_preview,
            self.worker_mic_encode_packet,
            self.worker_speaker_decode_packet,
            self.worker_grm_comm
        ]

        for i in range(len(threads)):
            if threads[i] is not None:
                threads[i].stop_process()

        check_continue: bool = True
        while check_continue is True:
            check_continue = False

            for i in range(len(threads)):
                if threads[i] is not None and threads[i].terminated is False:
                    check_continue = True
                    time.sleep(10)
                    break

        self.close()

    def on_change_use_reference_image(self):
        if self.checkBox_use_reference_image.isChecked() is True:
            self.lineEdit_search_reference_image.setDisabled(False)
            self.button_search_reference_image.setDisabled(False)
            self.worker_video_encode_packet.set_reference_image_frame(self.reference_image_frame)
        else:
            self.lineEdit_search_reference_image.setDisabled(True)
            self.button_search_reference_image.setDisabled(True)
            self.worker_video_encode_packet.set_reference_image_frame(None)

    def apply_reference_image(self, reference_image):
        try:
            with open(reference_image, "rb") as f:
                bytes_read = f.read()

                frame = np.frombuffer(bytes_read, dtype=np.uint8)
                frame = cv2.imdecode(frame, flags=1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                img = frame
                w, h = frame.shape[:2]
                if w != IMAGE_SIZE or h != IMAGE_SIZE:
                    x = 0
                    y = 0
                    if w > h:
                        x = int((w - h) / 2)
                        w = h
                    elif h > w:
                        y = int((h - w) / 2)
                        h = w

                    cropped_img = frame[x: x + w, y: y + h]
                    img = resize(cropped_img, (IMAGE_SIZE, IMAGE_SIZE))[..., :3]

                self.reference_image_frame = img
                img = resize(img, (self.reference_image_view.width(), self.reference_image_view.width()))[..., :3]

                h, w, c = img.shape
                q_img = QtGui.QImage(img.data, w, h, w * c, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(q_img)
                pixmap_resized = pixmap.scaledToWidth(self.reference_image_view.width())
                if pixmap_resized is not None:
                    self.reference_image_view.setPixmap(pixmap)
                    self.lineEdit_search_reference_image.setText(reference_image)

                    if self.checkBox_use_reference_image.isChecked() is True and \
                            self.worker_video_encode_packet is not None:
                        self.worker_video_encode_packet.set_reference_image_frame(img)

                    return True
        except Exception as err:
            print(err)

        return False

    def on_search_reference_image(self):
        reference_image = QFileDialog.getOpenFileName(self, filter='*.jpg')
        if reference_image is None or len(reference_image[0]) == 0:
            return

        self.apply_reference_image(reference_image[0])

    def on_user_selection_changed(self):
        selected_indexes = self.listWidget.selectedIndexes()
        if selected_indexes is None or len(selected_indexes) == 0:
            return

        selected_index = selected_indexes[0]
        self.listWidget.setCurrentRow(-1)

        selected_index_row = selected_index.row()
        if 0 <= selected_index_row < len(self.join_peer):
            peer_id = self.join_peer[selected_index_row].peer_id

            if peer_id is not None and self.worker_video_decode_and_render_packet is not None:
                self.worker_video_decode_and_render_packet.check_show_view(peer_id)

    def remove_room(self, show_alert:bool):
        ret: bool = False

        print(f"creationOverlayId:{self.join_session.creationOverlayId}, "
              f"ownerId:{owner_data.ownerPeerId}, "
              f"adminKey:{owner_data.ownerAdminKey}")
        if self.join_session.creationOverlayId is not None and \
                owner_data.ownerPeerId is not None and \
                owner_data.ownerAdminKey is not None:
            res = api.Removal(api.RemovalRequest(self.join_session.creationOverlayId,
                                                 owner_data.ownerPeerId,
                                                 owner_data.ownerAdminKey))
            if res.code is api.ResponseCode.Success:
                self.request_update_log('Succeed to delete the channel.')
                ret = True
            else:
                if show_alert is True:
                    self.request_alert('error', f'Failed to delete the channel. error code:{res.code}')

        self.create_button.setText("Channel Create")
        self.join_session = SessionData()
        return ret

    def output_chat(self, display_name, peer_id, message):
        if display_name is None and peer_id is not None:
            for i in self.join_peer:
                if i.peer_id == peer_id:
                    display_name = i.display_name
                    break

        print(f'output chat. display_name:{display_name} message:{message}')
        chat_message = '[' + display_name + '] : ' + message
        self.listWidget_chat_message.addItem(chat_message)

    def search_user(self):
        search_peer_req = api.SearchPeerRequest(self.join_session.overlayId)
        print("\nSearchPeerRequest:", search_peer_req)

        search_peer_res = api.SearchPeer(search_peer_req)
        print("\nSearchPeerResponse:", search_peer_res)
        # return searchPeerRes.peerList
        if search_peer_res.code is api.ResponseCode.Success:
            for i in search_peer_res.peerList:
                update_peer: PeerData = PeerData(peer_id=i.peerId, display_name=i.displayName)
                self.update_user(update_peer, False)

    def update_user(self, p_peer_data: PeerData, p_leave_flag: bool):
        if self.worker_video_decode_and_render_packet is not None:
            self.worker_video_decode_and_render_packet.update_user(p_peer_data, p_leave_flag)

        if self.worker_speaker_decode_packet is not None:
            self.worker_speaker_decode_packet.update_user(p_peer_data, p_leave_flag)

        if p_leave_flag is True:
            self.request_update_log(f'Leaved user. peerId:{p_peer_data.peer_id}')

            if self.join_peer is not None:
                for i in self.join_peer:
                    if p_peer_data.peer_id == i.peer_id:
                        self.join_peer.remove(p_peer_data)
                        break
        else:
            add_user = True
            if self.join_peer is not None:
                for i in self.join_peer:
                    if p_peer_data.peer_id == i.peer_id:
                        i.display_name = p_peer_data.display_name
                        add_user = False

            if add_user is True:
                self.request_update_log(f'Joined user. peerId:{p_peer_data.peer_id}')
                self.join_peer.append(p_peer_data)

                if self.worker_video_encode_packet is not None:
                    self.worker_video_encode_packet.request_send_key_frame()

    def update_user_list(self):
        self.listWidget.clear()
        for i in self.join_peer:
            self.listWidget.addItem(i.display_name)

    def session_notification_listener(self, change: api.Notification):
        # print(f"session_notification_listener notification.{change}")

        if change.notificationType is api.NotificationType.SessionChangeNotification:
            session_change: api.SessionChangeNotification = change
            print("SessionChangeNotification received.", session_change)
            print(f"Change session is {session_change.overlayId}")
            self.join_session = SessionData(overlayId=session_change.overlayId,
                                            title=session_change.title,
                                            description=session_change.title,
                                            ownerId=session_change.ownerId,
                                            accessKey=session_change.accessKey,
                                            sourceList=session_change.sourceList,
                                            channelList=session_change.channelList)
        elif change.notificationType is api.NotificationType.SessionTerminationNotification:
            self.request_update_log(f'Received the channel terminated')

            session_termination: api.SessionTerminationNotification = change
            print(f"SessionTerminationNotification received. {session_termination}")
            print(f"Terminate session is {session_termination.overlayId}")
            self.request_terminated_room()
        elif change.notificationType is api.NotificationType.PeerChangeNotification:
            peer_change: api.PeerChangeNotification = change
            print(f"PeerChangeNotification received. {peer_change}")

            peer_id = peer_change.changePeerId.split(';')[0]
            if self.join_session.overlayId == peer_change.overlayId:
                update_peer_data: PeerData = PeerData(peer_id=peer_id, display_name=peer_change.displayName)
                self.update_user(update_peer_data, peer_change.leave)
                self.update_user_list()
        elif change.notificationType is api.NotificationType.DataNotification:
            data: api.DataNotification = change
            if data.dataType == api.DataType.Data.value:
                peerId = data.sendPeerId.split(';')[0]
                # print(f"\nData received. peer_id:{peerId}")
                _, _, _, _, _mediatype, _, _bindata = self.bin_wrapper.parse_wrap_common_header(data.data)
                if _mediatype == TYPE_INDEX.TYPE_VIDEO:
                    media_queue_data = MediaQueueData(peerId, _bindata)
                    if self.worker_grm_comm is not None:
                        self.worker_grm_comm.recv_video_queue.put(media_queue_data)
                elif _mediatype == TYPE_INDEX.TYPE_AUDIO:
                    media_queue_data = MediaQueueData(peerId, _bindata)
                    if self.worker_grm_comm is not None:
                        self.worker_grm_comm.recv_audio_queue.put(media_queue_data)
                elif _mediatype == TYPE_INDEX.TYPE_DATA:
                    _type, _value, _ = self.bin_wrapper.parse_bin(_bindata)
                    if _type == TYPE_INDEX.TYPE_DATA_CHAT:
                        chat_message = self.bin_wrapper.parse_chat(_value)
                        print(f"chat_message. peer_id:{peerId} message:{chat_message}")
                        self.output_chat(None, peerId, chat_message)
