import time
import threading

import torch
import hp2papi as api

from GUI.MainWindow import MainWindowClass
from gooroomee.grm_defs import GrmParentThread, MediaQueueData
from gooroomee.grm_packet import BINWrapper, TYPE_INDEX
from gooroomee.grm_queue import GRMQueue


class GrmCommWorker(GrmParentThread):
    def __init__(self,
                 main_window,
                 p_send_audio_queue,
                 p_send_video_queue,
                 p_send_chat_queue,
                 p_recv_audio_queue,
                 p_recv_video_queue,
                 p_set_connect):
        super().__init__()
        # self.main_windows: MainWindowClass = p_main_windows
        self.main_window: MainWindowClass = main_window
        self.comm_bin = None
        self.client_connected: bool = False
        # self.lock = None
        self.sent_key_frame = False
        self.send_audio_queue: GRMQueue = p_send_audio_queue
        self.send_video_queue: GRMQueue = p_send_video_queue
        self.send_chat_queue: GRMQueue = p_send_chat_queue
        self.recv_audio_queue: GRMQueue = p_recv_audio_queue
        self.recv_video_queue: GRMQueue = p_recv_video_queue
        self.avatar = None
        self.kp_source = None
        self.avatar_kp = None
        self.bin_wrapper = BINWrapper()
        self.set_connect = p_set_connect
        self.stop_comm_request = False
        self.stop_comm_completed = False
        self.lock = threading.Lock()
        self.send_video_fps = 0

    def on_client_connected(self):
        print('grm_worker:on_client_connected')
        # self.lock.acquire()
        self.client_connected = True
        self.sent_key_frame = False
        self.set_connect(True)
        # self.lock.release()

    def on_client_closed(self):
        print('grm_worker:on_client_closed')
        # self.lock.acquire()
        self.client_connected = False
        self.sent_key_frame = False
        # self.set_join(False)
        # self.main_windows.set_connect(False)
        # self.lock.release()

    def on_client_data(self, bin_data):
        if self.client_connected is False:
            self.client_connected = True
            self.set_connect(True)
        if self.join_flag is False:
            return

        _version, _timestamp, _seq_num, _ssrc, _mediatype, _bindata_len, _bindata = self.bin_wrapper.parse_wrap_common_header(
            bin_data)
        if _mediatype == TYPE_INDEX.TYPE_VIDEO:
            media_queue_data = MediaQueueData("", _bindata)
            self.recv_video_queue.put(media_queue_data)
        elif _mediatype == TYPE_INDEX.TYPE_AUDIO:
            media_queue_data = MediaQueueData("", _bindata)
            self.recv_audio_queue.put(media_queue_data)
        elif _mediatype == TYPE_INDEX.TYPE_DATA:
            _type, _value, _ = self.bin_wrapper.parse_bin(_bindata)
            if _type == TYPE_INDEX.TYPE_DATA_CHAT:
                chat_message = self.bin_wrapper.parse_chat(_value)
                print(f"chat_message : {chat_message}")
                self.main_window.output_chat(chat_message)

    def set_join(self, p_join_flag: bool):
        GrmParentThread.set_join(self, p_join_flag)

        self.send_audio_queue.clear()
        self.send_video_queue.clear()
        self.send_chat_queue.clear()

        if p_join_flag is False:
            self.lock.acquire()
            self.stop_comm_request = True
            self.stop_comm_completed = False
            self.lock.release()

    def is_stopped_comm(self):
        ret: bool = False

        self.lock.acquire()
        if self.stop_comm_completed is True:
            self.stop_comm_completed = False
            ret = True
        self.lock.release()

        return ret

    def get_send_video_fps(self):
        send_video_fps = self.send_video_fps
        self.send_video_fps = 0
        return send_video_fps

    def run(self):
        while self.alive:
            print(f'GrmCommWorker running:{self.running}')
            while self.running:
                if self.join_flag is False:
                    self.lock.acquire()
                    if self.stop_comm_request is True:
                        self.stop_comm_request = False
                        self.stop_comm_completed = True
                    self.lock.release()

                    continue

                # print(f'GrmCommWorker queue size:{self.send_audio_queue.length()}')
                if self.send_audio_queue.length() > 0:
                    # print(f'GrmCommWorker pop queue size:{self.send_audio_queue.length()}')
                    audio_bin_data = self.send_audio_queue.pop()
                    if audio_bin_data is not None:
                        audio_channel_id = self.main_window.join_session.audio_channel_id()
                        if audio_channel_id is not None:
                            send_request = api.SendDataRequest(api.DataType.Data,
                                                               self.main_window.join_session.overlayId,
                                                               self.main_window.join_session.audio_channel_id(),
                                                               audio_bin_data)

                            # print("SendData Audio Request")
                            res = api.SendData(send_request)
                            # print("  SendData Audio Response")

                            if res.code is api.ResponseCode.Success:
                                # print("Video SendData success")
                                pass
                            else:
                                print("Audio SendData fail.", res.code)

                # print(f'GrmCommWorker queue size:{self.video_packet_queue.length()}')
                if self.send_video_queue.length() > 0:
                    video_bin_data = self.send_video_queue.pop()

                    if video_bin_data is not None:
                        video_channel_id = self.main_window.join_session.video_channel_id()
                        if video_channel_id is not None:
                            send_request = api.SendDataRequest(api.DataType.Data,
                                                               self.main_window.join_session.overlayId,
                                                               self.main_window.join_session.video_channel_id(),
                                                               video_bin_data)
                            # print("SendData Video Request")
                            res = api.SendData(send_request)
                            # print("SendData Video Response")

                            if res.code is api.ResponseCode.Success:
                                # print("Video SendData success")
                                self.send_video_fps += 1
                                pass
                            else:
                                print("Video SendData fail.", res.code)

                if self.send_chat_queue.length() > 0:
                    # print(f'GrmCommWorker pop queue size:{self.send_chat_queue.length()}')
                    chat_bin_data = self.send_chat_queue.pop()
                    if chat_bin_data is not None:
                        text_channel_id = self.main_window.join_session.text_channel_id()
                        if text_channel_id is not None:
                            send_request = api.SendDataRequest(api.DataType.Data,
                                                               self.main_window.join_session.overlayId,
                                                               self.main_window.join_session.text_channel_id(),
                                                               chat_bin_data)
                            # print("SendData Chat Request")
                            res = api.SendData(send_request)
                            # print("SendData Chat Response")

                            if res.code is api.ResponseCode.Success:
                                # print("Video SendData success")
                                pass
                            else:
                                print("Chat SendData fail.", res.code)

                time.sleep(0.001)
            time.sleep(0.001)

        print("Stop GrmCommWorker")
        self.terminated = True
        # self.terminate()

