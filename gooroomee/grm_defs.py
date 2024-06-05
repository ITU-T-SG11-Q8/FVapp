import hp2papi as api
import pyaudio
import time

from dataclasses import dataclass
from PyQt5.QtCore import QThread
from hp2papi.classes import Channel, ChannelType
from typing import List

IMAGE_SIZE = 384

# 음성 출력 설정
RATE = 44100
CHANNELS = 1
FORMAT = pyaudio.paInt16
SPK_CHUNK = 2 ** 14
MIC_CHUNK = 2 ** 14


class ModeType:
    SNNM = 'snnm'
    KDM = 'kdm'


ownerDataFileName = 'config.ini'


class OwnerData:
    ownerPeerId: str = None
    ownerAdminKey: str = None
    ownerUseFaceVideo: bool = None
    ownerUseAudio: bool = None
    ownerUseText: bool = None
    ownerModeType: ModeType = None
    ownerDisplayName: str = None
    ownerPrivateKey: str = None

    def __init__(self):
        self.read_values()

    def read_values(self):
        self.ownerPeerId = ''
        self.ownerAdminKey = ''
        self.ownerUseFaceVideo = True
        self.ownerUseAudio = False
        self.ownerUseText = True
        self.ownerModeType = ModeType.SNNM
        self.ownerDisplayName = ''
        self.ownerPrivateKey = ''

        try:
            with open(ownerDataFileName, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    sub_lines = line.split('=')
                    if len(sub_lines) == 2:
                        l_value = sub_lines[0].strip()
                        r_value = sub_lines[1].strip()

                        if l_value == 'PEER_ID':
                            self.ownerPeerId = r_value
                        elif l_value == 'ADMIN_KEY':
                            self.ownerAdminKey = r_value
                        elif l_value == 'USE_FACE_VIDEO':
                            if r_value == 'True':
                                self.ownerUseFaceVideo = True
                            else:
                                self.ownerUseFaceVideo = False
                        elif l_value == 'USE_AUDIO':
                            if r_value == 'True':
                                self.ownerUseAudio = True
                            else:
                                self.ownerUseAudio = False
                        elif l_value == 'USE_TEXT':
                            if r_value == 'True':
                                self.ownerUseText = True
                            else:
                                self.ownerUseText = False
                        elif l_value == 'MODE_TYPE':
                            if r_value == ModeType.KDM:
                                self.ownerModeType = ModeType.KDM
                            else:
                                self.ownerModeType = ModeType.SNNM
                        elif l_value == 'DISPLAY_NAME':
                            self.ownerDisplayName = r_value
                        elif l_value == 'PRIVATE_KEY':
                            self.ownerPrivateKey = r_value

        except IOError:
            print(f'Could not read file:{ownerDataFileName}')

    def write_values(self):
        lines = [f'PEER_ID={self.ownerPeerId}',
                 f'ADMIN_KEY={self.ownerAdminKey}',
                 f'USE_FACE_VIDEO={self.ownerUseFaceVideo}',
                 f'USE_AUDIO={self.ownerUseAudio}',
                 f'USE_TEXT={self.ownerUseText}',
                 f'MODE_TYPE={self.ownerModeType}',
                 f'DISPLAY_NAME={self.ownerDisplayName}',
                 f'PRIVATE_KEY={self.ownerPrivateKey}']

        try:
            with open(ownerDataFileName, 'w') as file:
                for line in lines:
                    file.write(line)
                    file.write('\n')

        except IOError:
            print(f'Could not read file:{ownerDataFileName}')

    def set_value_peer_id(self, peer_id):
        self.ownerPeerId = peer_id

    def set_value_admin_key(self, admin_key):
        self.ownerAdminKey = admin_key

    def set_value_use_face_video(self, use_face_video):
        self.ownerUseFaceVideo = use_face_video

    def set_value_use_audio(self, use_audio):
        self.ownerUseAudio = use_audio

    def set_value_use_text(self, use_text):
        self.ownerUseText = use_text

    def set_value_mode_type(self, mode_type):
        self.ownerModeType = mode_type

    def set_value_display_name(self, display_name):
        self.ownerDisplayName = display_name

    def set_value_private_key(self, private_key):
        self.ownerPrivateKey = private_key


# @dataclass
class SessionData:
    creationOverlayId: str = None
    creationTitle: str = None

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

    def set_join(self, p_join_flag: bool):
        # print(f'set_join join_flag = [{p_join_flag}]')
        self.join_flag = p_join_flag
