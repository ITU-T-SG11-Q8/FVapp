import os

import pyaudio
from PyQt5 import QtCore, uic, QtGui
from PyQt5.QtCore import QTimer, pyqtSlot, QThread
from PyQt5.QtWidgets import QMainWindow, QFileDialog

from GUI.RoomCreate import RoomCreateClass
from GUI.RoomInformation import RoomInformationClass
from GUI.RoomJoin import RoomJoinClass
from afy.arguments import opt
from gooroomee.grm_defs import ModeType, SessionData, PeerData, IMAGE_SIZE, GrmParentThread, MediaQueueData
from gooroomee.grm_packet import BINWrapper, TYPE_INDEX
from typing import List
from afy.utils import crop, resize

import hp2papi as api
import time
import pygame
import pygame.camera
import cv2
import numpy as np

form_class = uic.loadUiType("GUI/MAIN_WINDOW.ui")[0]


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


class ThreadUpdateLog(QThread):
    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window
        self.running = False

    def __del__(self):
        self.wait()

    def start_process(self):
        self.running = True
        self.start()

    def stop_process(self):
        self.running = False

    def run(self):
        time_start = get_current_time_ms()

        while self.running:
            if get_current_time_ms() - time_start >= 1000:
                time_start = get_current_time_ms()
                self.main_window.request_update_stat('')

        time.sleep(0.01)


class MainWindowClass(QMainWindow, form_class):
    mode_type: ModeType = ModeType.SNNM
    replace_image_frame = None
    update_log_signal = QtCore.pyqtSignal(str)
    update_stat_signal = QtCore.pyqtSignal(str)
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

        self.camera_device_init(5)
        self.audio_device_init()

        self.keyframe_period = opt.keyframe_period
        if self.keyframe_period is None:
            self.keyframe_period = 0

        print(f'key frame period:{self.keyframe_period}')

        self.send_chat_queue = None                         # GRMQueue
        self.worker_capture_frame = None                    # CaptureFrameWorker
        self.worker_video_encode_packet = None              # EncodeVideoPacketWorker
        self.worker_video_decode_and_render_packet = None   # DecodeAndRenderVideoPacketWorker
        self.worker_speaker_decode_packet = None            # DecodeSpeakerPacketWorker
        self.worker_grm_comm = None                         # GrmCommWorker

        self.get_worker_seq_num = p_get_worker_seq_num
        self.get_worker_ssrc = p_get_worker_ssrc
        self.set_join = p_set_join

        self.create_button.clicked.connect(self.create_room)
        self.join_button.clicked.connect(self.join_room)
        self.room_information_button.clicked.connect(self.information_room)
        self.button_chat_send.clicked.connect(self.send_chat)
        self.lineEdit_input_chat.returnPressed.connect(self.send_chat)
        self.comboBox_mic.currentIndexChanged.connect(self.change_mic_device)
        self.comboBox_audio_device.currentIndexChanged.connect(self.change_audio_device)
        self.comboBox_video_device.currentIndexChanged.connect(self.change_camera_device)
        self.button_exit.clicked.connect(self.exit_button)
        self.checkBox_use_replace_image.stateChanged.connect(self.change_use_replace_image)
        self.button_search_replace_image.clicked.connect(self.search_replace_image)

        self.button_chat_send.setDisabled(True)
        self.lineEdit_input_chat.setDisabled(True)
        self.overlay_id = ""
        self.peer_id = ""
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

        self.update_log_signal.connect(self.update_log)
        self.update_stat_signal.connect(self.update_stat)

        self.thread_update_log = ThreadUpdateLog(self)
        self.thread_update_log.start_process()

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

    def request_update_stat(self, stat):
        video_capture_queue_count = 0
        video_send_queue_count = 0
        video_recv_queue_count = 0

        if self.worker_video_encode_packet and self.worker_video_encode_packet.video_capture_queue:
            video_capture_queue_count = self.worker_video_encode_packet.video_capture_queue.length()
        if self.worker_grm_comm and self.worker_grm_comm.send_video_queue:
            video_encode_and_send_queue_count = self.worker_grm_comm.send_video_queue.length()
        if self.worker_grm_comm and self.worker_grm_comm.recv_video_queue:
            video_recv_queue_count = self.worker_grm_comm.recv_video_queue.length()

        stat = f'capture video queue count : {video_capture_queue_count}\nsend video encode and send queue count : {video_encode_and_send_queue_count}\nrecv video queue count : {video_recv_queue_count}'

        self.update_stat_signal.emit(stat)

    def set_workers(self,
                    p_send_chat_queue,
                    p_worker_capture_frame,
                    p_worker_video_encode_packet,
                    p_worker_video_decode_and_render_packet,
                    p_worker_speaker_decode_packet,
                    p_worker_grm_comm):
        self.send_chat_queue = p_send_chat_queue                                                # GRMQueue
        self.worker_capture_frame = p_worker_capture_frame                                      # CaptureFrameWorker
        self.worker_video_encode_packet = p_worker_video_encode_packet                          # EncodeVideoPacketWorker
        self.worker_video_decode_and_render_packet = p_worker_video_decode_and_render_packet    # DecodeAndRenderVideoPacketWorker
        self.worker_speaker_decode_packet = p_worker_speaker_decode_packet                      # DecodeSpeakerPacketWorker
        self.worker_grm_comm = p_worker_grm_comm                                                # GrmCommWorker

        if self.worker_video_encode_packet is not None:
            self.button_send_keyframe.clicked.connect(self.worker_video_encode_packet.request_send_key_frame)
            self.button_request_keyframe.clicked.connect(self.worker_video_encode_packet.request_recv_key_frame)

    def timeout_key_frame(self):
        if self.worker_video_encode_packet is not None:
            self.worker_video_encode_packet.request_send_key_frame()

    def camera_device_init(self, max_count):
        pygame.camera.init()
        # make the list of all available cameras
        cam_list = pygame.camera.list_cameras()
        print(f"camlist:{cam_list}")
        for index in cam_list:
            device_string = "Camera #" + str(index)
            self.comboBox_video_device.addItem(device_string, userData=index)

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
                        self.comboBox_audio_device.addItem(device_name, userData=index)
                if key == "maxOutputChannels":
                    if audio_info[key] == 0:
                        print(f"Output deviceName:{device_name}, index:{index}")
                        self.comboBox_mic.addItem(device_name, userData=index)

    def create_room(self):
        if self.create_button.text() == "Channel Create":
            self.room_create_ui.clear_value()
            self.room_create_ui.show()

        elif self.create_button.text() == "Channel Delete":
            self.remove_room()

    def create_room_ok_func(self):
        if self.create_button.text() == "Channel Create":
            title = self.room_create_ui.lineEdit_title.text()
            description = self.room_create_ui.lineEdit_description.text()
            owner_id = self.room_create_ui.lineEdit_ower_id.text()
            admin_key = self.room_create_ui.lineEdit_admin_key.text()
            checked_face_video = self.room_create_ui.checkBox_facevideo.isChecked()
            checked_audio = self.room_create_ui.checkBox_audio.isChecked()
            checked_text = self.room_create_ui.checkBox_text.isChecked()

            if self.room_create_ui.radioButton_kdm.isChecked() is True:
                self.mode_type = ModeType.KDM
                self.request_update_log(f'request to create KDM room. title:{title}')
            else:
                self.mode_type = ModeType.SNNM
                self.request_update_log(f'request to create SNNM mode room. title:{title}')

            creation_req = api.CreationRequest(title=title, description=description,
                                               ownerId=owner_id, adminKey=admin_key)

            service_control_channel = api.ChannelServiceControl()
            service_control_channel.channelId = "controlChannel"
            face_video_channel = None
            audio_channel = None
            text_channel = None

            if checked_face_video is True:
                face_video_channel = internal_create_channel_facevideo(self.mode_type)
            if checked_audio is True:
                audio_channel = internal_create_channel_audio()
            if checked_text is True:
                text_channel = internal_create_channel_text()

            creation_req.channelList = [service_control_channel, face_video_channel, audio_channel, text_channel]

            print("\nCreationRequest:", creation_req)
            creation_req.sourceList = ["*"]
            creation_res = api.Creation(creation_req)
            self.room_create_ui.close()

            print("\nCreationResponse:", creation_res)

            if creation_res.code is api.ResponseCode.Success:
                print("\nCreation success.", creation_res.overlayId)
                if self.room_create_ui.radioButton_kdm.isChecked() is True:
                    self.request_update_log(f'succeed to create KDM mode room. overlayId:{creation_res.overlayId}')
                else:
                    self.request_update_log(f'succeed to create SNNM mode room. overlayId:{creation_res.overlayId}')

                self.join_session.creationOverlayId = creation_res.overlayId
                self.join_session.creationTitle = title
                self.join_session.creationOwnerId = owner_id
                self.join_session.creationAdminKey = admin_key

                self.join_session.overlayId = creation_res.overlayId
                self.join_session.ownerId = owner_id
                self.join_session.adminKey = admin_key

                self.create_button.setText("Channel Delete")
                self.room_information_button.setDisabled(False)

                api.SetNotificatonListener(self.join_session.overlayId, self.join_session.ownerId,
                                           func=self.session_notification_listener)
            else:
                print("\nCreation fail.", creation_res.code)
                if self.mode_type == ModeType.KDM:
                    self.request_update_log(f'failed to create KDM mode room. overlayId:{creation_res.overlayId}')
                else:
                    self.request_update_log(f'failed to create SNNM mode room. overlayId:{creation_res.overlayId}')

                self.join_session.overlayId = ""

        elif self.create_button.text() == "Channel Delete":
            self.request_update_log(f'delete room.')

            self.remove_room()
            self.all_stop_worker()
            self.room_create_ui.close()

    def join_room(self):
        if self.join_button.text() == "Channel Join":
            self.join_ui.clear_value()

            if self.join_session.creationOverlayId is None or len(self.join_session.creationOverlayId) == 0:
                self.join_ui.button_query.setDisabled(False)
                self.join_ui.comboBox_overlay_id.setDisabled(False)
            else:
                self.join_ui.button_query.setDisabled(True)
                self.join_ui.comboBox_overlay_id.setDisabled(True)
                self.join_ui.comboBox_overlay_id.addItem(self.join_session.creationOverlayId)
                self.join_ui.lineEditTitle.setText(self.join_session.creationTitle)

            if self.join_session.creationOwnerId is None or len(self.join_session.creationOwnerId) == 0:
                self.join_ui.lineEdit_peer_id.setText('')
                self.join_ui.lineEdit_peer_id.setReadOnly(False)
            else:
                self.join_ui.lineEdit_peer_id.setText(self.join_session.creationOwnerId)
                self.join_ui.lineEdit_peer_id.setReadOnly(True)

            self.join_ui.show()
        elif self.join_button.text() == "Channel Leave":
            self.leave_room()

    def send_join_room_func(self):
        if self.join_button.text() == "Channel Join":
            self.overlay_id = self.join_ui.comboBox_overlay_id.currentText()
            self.peer_id = self.join_ui.lineEdit_peer_id.text()
            self.display_name = self.join_ui.lineEdit_display_name.text()
            self.private_key = self.join_ui.lineEdit_private_key.text()

            if len(self.private_key) == 0:
                self.request_update_log(f'no private_key find.')
                return

            self.join_session.overlayId = self.overlay_id

            self.room_information_button.setDisabled(False)
            self.create_button.setDisabled(True)

            self.button_chat_send.setDisabled(False)
            self.lineEdit_input_chat.setDisabled(False)

            self.join_session.overlayId = self.overlay_id
            self.join_session.ownerId = self.peer_id
            self.room_information_button.setDisabled(False)

            api.SetNotificatonListener(overlayId=self.join_session.overlayId,
                                       peerId=self.join_session.ownerId,
                                       func=self.session_notification_listener)

            self.request_update_log(f'request to join room')
            private_key_abs = os.path.abspath(self.private_key)
            join_request = api.JoinRequest(self.overlay_id, "", self.peer_id, self.display_name, private_key_abs)
            print("\nJoinRequest:", join_request)
            join_response = api.Join(join_request)
            print("\nJoinResponse:", join_response)
            self.join_ui.close()

            if join_response.code is api.ResponseCode.Success:
                self.join_button.setText("Channel Leave")
                self.join_session.channelList = join_response.channelList
                self.mode_type = self.join_session.video_channel_type()
                self.join_session.title = join_response.title
                self.join_session.description = join_response.description
                self.set_join(True)

                if self.mode_type == ModeType.KDM:
                    self.request_update_log(f'succeed to join KDM mode room')
                else:
                    self.request_update_log(f'succeed to join SNNM mode room')

                if self.worker_video_encode_packet is not None:
                    self.worker_video_encode_packet.request_send_key_frame()
            else:
                if self.mode_type == ModeType.KDM:
                    self.request_update_log(f'failed to join KDM mode room. code:{join_response.code}')
                else:
                    self.request_update_log(f'failed to join SNNM mode room. code:{join_response.code}')

            return join_response
        elif self.join_button.text() == "Channel Leave":
            self.leave_room()

    def leave_room(self):
        self.request_update_log(f'request to leave room.')

        self.set_join(False)
        self.join_button.setText("Channel Join")
        self.create_button.setDisabled(False)

        while self.worker_grm_comm.is_stopped_comm() is False:
            time.sleep(1)

        res = api.Leave(api.LeaveRequest(overlayId=self.join_session.overlayId,
                                         peerId=self.peer_id,
                                         accessKey=self.join_session.accessKey))

        if res.code is api.ResponseCode.Success:
            self.request_update_log(f'succeed to leave room.')
        else:
            self.request_update_log(f'failed to leave room. code:{res.code}')
            print("\nLeave fail.", res.code)

        self.join_session.clear_value()
        self.listWidget.clear()
        self.listWidget_chat_message.clear()

    def information_room(self):
        if self.join_session.overlayId is not None:
            self.room_information_ui.lineEdit_overlay_id.setText(self.join_session.overlayId)
        self.room_information_ui.lineEdit_overlay_id.setDisabled(True)

        if self.join_session.ownerId is not None:
            self.room_information_ui.lineEdit_ower_id.setText(self.join_session.ownerId)
        self.room_information_ui.lineEdit_ower_id.setDisabled(True)

        if self.join_session.accessKey is not None:
            self.room_information_ui.lineEdit_admin_key.setText(self.join_session.accessKey)
        self.room_information_ui.lineEdit_admin_key.setDisabled(True)

        if self.join_session.title is not None:
            self.room_information_ui.lineEdit_title.setText(self.join_session.title)

        if self.join_session.description is not None:
            self.room_information_ui.lineEdit_description.setText(self.join_session.description)

        if self.join_session.adminKey is not None and len(self.join_session.adminKey) > 0:
            self.room_information_ui.button_ok.setDisabled(False)
        else:
            self.room_information_ui.button_ok.setDisabled(True)

        self.room_information_ui.groupBox.setCheckable(False)
        self.room_information_ui.checkBox_facevideo.setChecked(False)
        self.room_information_ui.checkBox_audio.setChecked(False)
        self.room_information_ui.checkBox_text.setChecked(False)

        if self.join_session.channelList is not None:
            for i in self.join_session.channelList:
                if i.channelType is api.ChannelType.FeatureBasedVideo:
                    self.room_information_ui.checkBox_facevideo.setChecked(True)
                    self.room_information_ui.checkBox_facevideo.setDisabled(True)
                elif i.channelType is api.ChannelType.Audio:
                    self.room_information_ui.checkBox_audio.setChecked(True)
                    self.room_information_ui.checkBox_audio.setDisabled(True)
                elif i.channelType is api.ChannelType.Text:
                    self.room_information_ui.checkBox_text.setChecked(True)
                    self.room_information_ui.checkBox_text.setDisabled(True)
        self.room_information_ui.show()

    def modify_information_room(self):
        self.request_update_log(f'request to modify information room')
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

        face_video_channel = None
        audio_channel = None
        text_channel = None

        if self.room_information_ui.checkBox_facevideo.isChecked():
            face_video_channel = internal_create_channel_facevideo(self.mode_type)
            face_video_channel.sourceList = ["*"]
        if self.room_information_ui.checkBox_audio.isChecked():
            audio_channel = internal_create_channel_audio()
            audio_channel.sourceList = ["*"]
        if self.room_information_ui.checkBox_text.isChecked():
            text_channel = internal_create_channel_text()
            text_channel.sourceList = ["*"]
        if face_video_channel is not None or audio_channel is not None or text_channel is not None:
            modification_req.channelList = [face_video_channel, audio_channel, text_channel]

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
            self.request_update_log(f'succeed to modify information room')
            print("\nModification success.")
        else:
            self.request_update_log(f'failed to modify information room')
            print("\nModification fail.", modification_res.code)

        self.room_information_ui.close()

    def send_chat(self):
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

    def change_mic_device(self):
        self.worker_mic_encode_packet.pause_process()
        time.sleep(2)
        self.worker_mic_encode_packet.change_device(self.comboBox_mic.currentData())
        # self.worker_mic_encode_packet.change_device(1)
        self.worker_mic_encode_packet.resume_process()

    def change_audio_device(self):
        if self.worker_speaker_decode_packet is not None:
            print('main change speaker device start')
            self.worker_speaker_decode_packet.pause_process()
            time.sleep(2)
            self.worker_speaker_decode_packet.change_device(self.comboBox_audio_device.currentData())
            self.worker_speaker_decode_packet.resume_process()
            print('main change speaker device end')

    def change_camera_device(self):
        print(f'camera index change. {self.comboBox_video_device.currentData()}')
        if self.worker_capture_frame is not None:
            self.worker_capture_frame.pause_process()
            time.sleep(1)
            self.worker_capture_frame.change_device(self.comboBox_video_device.currentData())
            self.worker_capture_frame.resume_process()

    def exit_button(self):
        self.thread_update_log.stop_process()

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

    def change_use_replace_image(self):
        if self.checkBox_use_replace_image.isChecked() is True:
            self.lineEdit_search_replace_image.setDisabled(False)
            self.button_search_replace_image.setDisabled(False)
            self.worker_video_encode_packet.set_replace_image_frame(self.replace_image_frame)
        else:
            self.lineEdit_search_replace_image.setDisabled(True)
            self.button_search_replace_image.setDisabled(True)
            self.worker_video_encode_packet.set_replace_image_frame(None)

    def search_replace_image(self):
        replace_image = QFileDialog.getOpenFileName(self, filter='*.jpg')
        if replace_image is None or len(replace_image[0]) == 0:
            return

        self.lineEdit_search_replace_image.setText(replace_image[0])

        try:
            with open(replace_image[0], "rb") as f:
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

                self.replace_image_frame = img

                if self.checkBox_use_replace_image.isChecked() is True and self.worker_video_encode_packet is not None:
                    self.worker_video_encode_packet.set_replace_image_frame(self.replace_image_frame)

                h, w, c = self.replace_image_frame.shape
                q_img = QtGui.QImage(self.replace_image_frame.data, w, h, w * c, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(q_img)
                pixmap_resized = pixmap.scaledToWidth(self.replace_image_view.width())
                if pixmap_resized is not None:
                    self.replace_image_view.setPixmap(pixmap)
        except Exception as err:
            print(err)

    def remove_room(self):
        print(f"overlayId:{self.join_session.creationOverlayId}, ownerId:{self.join_session.creationOwnerId}, "
              f"adminKey:{self.join_session.creationAdminKey}")
        if self.join_session.creationOverlayId is not None and self.join_session.creationOwnerId is not None and self.join_session.creationAdminKey is not None:
            res = api.Removal(api.RemovalRequest(self.join_session.creationOverlayId, self.join_session.creationOwnerId,
                                                 self.join_session.creationAdminKey))
            if res.code is not api.ResponseCode.Success:
                print(f"\nRemoval fail.[{res.code}]")

        self.create_button.setText("Channel Create")
        print("\nRemoval success.")
        self.join_session = SessionData()

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
            self.request_update_log(f'leaved user. peerId:{p_peer_data.peer_id}')

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
                self.request_update_log(f'joined user. peerId:{p_peer_data.peer_id}')
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
            self.request_update_log(f'received session termination')

            session_termination: api.SessionTerminationNotification = change
            print(f"SessionTerminationNotification received. {session_termination}")
            print(f"Terminate session is {session_termination.overlayId}")
            self.leave_room()
            self.remove_room()
        elif change.notificationType is api.NotificationType.PeerChangeNotification:
            peer_change: api.PeerChangeNotification = change
            print(f"PeerChangeNotification received. {peer_change}")
            print(f"Peer change session is {peer_change.overlayId}")

            peerId = peer_change.changePeerId.split(';')[0]
            if self.join_session.overlayId == peer_change.overlayId:
                update_peer_data: PeerData = PeerData(peer_id=peerId, display_name=peer_change.displayName)
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

