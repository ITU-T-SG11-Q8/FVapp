import hp2papi as api
import pyaudio
import time

from dataclasses import dataclass
from PyQt5.QtCore import QThread
from hp2papi.classes import Channel, ChannelType
from typing import List

IMAGE_SIZE = 256

# 음성 출력 설정
RATE = 44100
CHANNELS = 1
FORMAT = pyaudio.paInt16
SPK_CHUNK = 2 ** 14
MIC_CHUNK = 2 ** 14


class ModeType:
    SNNM = 'snnm'
    KDM = 'kdm'


# @dataclass
class SessionData:
    creationOverlayId: str = None
    creationTitle: str = None
    creationOwnerId: str = None
    creationAdminKey: str = None

    adminKey: str = None
    overlayId: str = None
    title: str = None
    description: str = None
    startDateTime: str = None
    endDateTime: str = None
    ownerId: str = None
    accessKey: str = None
    sourceList: List[str] = None
    # channelList: List[peerApi.classes.Channel] = None
    channelList: List[Channel] = None

    def __init__(self):
        self.clear_value()

    def clear_value(self):
        self.adminKey = ''
        self.overlayId = ''
        self.title = ''
        self.description = ''
        self.startDateTime = ''
        self.endDateTime = ''
        self.ownerId = ''
        self.accessKey = ''
        self.sourceList = List[str]
        # self.channelList: List[peerApi.classes.Channel] = []
        # self.channelList: List[Channel] = []

    def audio_channel_id(self):
        if self.channelList is not None:
            for i in range(len(self.channelList)):
                if self.channelList[i].channelType is ChannelType.Audio:
                    return self.channelList[i].channelId
        return None

    def video_channel_id(self):
        if self.channelList is not None:
            for i in range(len(self.channelList)):
                if self.channelList[i].channelType is ChannelType.FeatureBasedVideo:
                    return self.channelList[i].channelId
        return None

    def video_channel_type(self):
        if self.channelList is not None:
            for i in range(len(self.channelList)):
                if self.channelList[i].channelType is ChannelType.FeatureBasedVideo:
                    if self.channelList[i].mode == api.FeatureBasedVideoMode.KeypointsDescriptionMode:
                        return ModeType.KDM
                    break
        return ModeType.SNNM

    def text_channel_id(self):
        if self.channelList is not None:
            for i in range(len(self.channelList)):
                if self.channelList[i].channelType is ChannelType.Text:
                    return self.channelList[i].channelId
        return None


@dataclass
class MediaQueueData:
    def __init__(self, peer_id, bin_data):
        self.peer_id = peer_id
        self.bin_data = bin_data


@dataclass
class PeerData:
    peer_id: str
    display_name: str


class GrmParentThread(QThread):
    def __init__(self):
        super().__init__()
        self.alive: bool = True
        self.running: bool = False
        self.terminated: bool = False
        self.device_index: int = 0
        self.join_flag: bool = False

    def __del__(self):
        self.wait()

    def start_process(self):
        self.running = True
        self.start()

    def stop_process(self):
        self.alive = False
        self.running = False

    def pause_process(self):
        self.running = False

    def resume_process(self):
        self.running = True

    def change_device(self, p_device_index):
        self.running = False
        print(f'change device index = [{p_device_index}]')
        self.device_index = p_device_index
        time.sleep(2)
        self.running = True

    def set_join(self, p_join_flag: bool):
        # print(f'set_join join_flag = [{p_join_flag}]')
        self.join_flag = p_join_flag
