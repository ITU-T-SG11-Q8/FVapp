from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import List

__all__ = [
    "ResponseCode",
    "DataType",
    "ChannelType",
    "FeatureBasedVideoMode",
    "AudioCodec",
    "AudioSampleRate",
    "AudioBitrate",
    "AudioMono",
    "TextFormat",
    "Response",
    "OverlayClosed",
    "Overlay",
    "QueryResponse",
    "ChannelServiceControl",
    "ChannelFeatureBasedVideo",
    "ChannelAudio",
    "ChannelText",
    "Request",
    "CreationRequest",
    "CreationResponse",
    "ModificationRequest",
    "JoinRequest",
    "JoinResponse",
    "Peer",
    "SearchPeerRequest",
    "SearchPeerResponse",
    "SendDataRequest",
    "LeaveRequest",
    "RemovalRequest",
    "Notification",
    "NotificationType",
    "SessionChangeNotification",
    "SessionTerminationNotification",
    "PeerChangeNotification",
    "DataNotification"
]

@unique
class ResponseCode(Enum):
    Success = 200
    WrongRequest = 400
    AuthenticationError = 403
    NotFound = 404
    Fail = 500

@unique
class DataType(Enum):
    FeatureBasedVideo = "feature/face"
    Audio = "audio"
    Text = "text"

@unique
class ChannelType(Enum):
    ServiceControl = "control"
    FeatureBasedVideo = "feature/face"
    Audio = "audio"
    Text = "text"

@unique
class FeatureBasedVideoMode(Enum):
    KeypointsDescriptionMode = "KDM"
    SharedNeuralNetworkMode = "SNNM"

@unique
class AudioCodec(Enum):
    AAC = "AAC"
    ALAC = "ALAC"
    AMR = "AMR"
    FLAC = "FLAC"
    G711 = "G711"
    G722 = "G722"
    MP3 = "MP3"
    Opus = "Opus"
    Vorbis = "Vorbis"

@unique
class AudioSampleRate(Enum):
    Is44100 = "44100"
    Is48000 = "48000"

@unique
class AudioBitrate(Enum):
    Is128kbps = "128kbps"
    Is192kbps = "192kbps"
    Is256kbps = "256kbps"
    Is320kbps = "320kbps"

@unique
class AudioMono(Enum):
    Mono = "mono"
    Stereo = "stereo"

@unique
class TextFormat(Enum):
    Plain = "Plain"
    Json = "JSON"

@unique
class OverlayClosed(Enum):
    Open = 0
    SetAccessKey = 1
    SetPeerList = 2

@unique
class NotificationType(Enum):
    SessionChangeNotification = 0
    SessionTerminationNotification = 1
    PeerChangeNotification = 2
    DataNotification = 3

def _check(var) -> bool:
    if not var:
        return False
    
    return True

@dataclass
class Request(metaclass=ABCMeta):
    @abstractmethod
    def mandatoryCheck(self) -> bool:
        pass

@dataclass
class Response(metaclass=ABCMeta):
    code: ResponseCode = None

@dataclass
class Channel:
    channelId: str = None
    channelType: ChannelType = None
    sourceList: List[str] = None

    def mandatoryCheck(self) -> bool:
        if not _check(self.channelType):
            return False
        
        return True

@dataclass
class Overlay:
    overlayId: str = None
    title: str = None
    description: str = None
    ownerId: str = None
    closed: OverlayClosed = OverlayClosed.Open

@dataclass
class QueryResponse(Response):
    overlay: List[Overlay] = None

@dataclass
class ChannelServiceControl(Channel):
    """
    제어 신호 메시지를 위한 채널
    """
    def __init__(self):
        self.channelType = ChannelType.ServiceControl
    
    def mandatoryCheck(self) -> bool:
        if not super().mandatoryCheck():
            return False
            
        return True

@dataclass
class ChannelFeatureBasedVideo(Channel):
    """
    feature를 위한 채널
    """
    def __init__(
        self,
        channelId: str = None,
        mode: FeatureBasedVideoMode = None,
        resolution: str = None,
        framerate: str = None,
        keypointsType: str = None,
        modelLink: str = None,
        hash: str = None,
        dimension: str = None,
    ):
        self.channelId = channelId
        self.channelType = ChannelType.FeatureBasedVideo
        self.mode = mode
        self.resolution = resolution
        self.framerate = framerate
        self.keypointsType = keypointsType
        self.modelLink = modelLink
        self.hash = hash
        self.dimension = dimension

    mode: FeatureBasedVideoMode = None
    resolution: str = None
    framerate: str = None
    keypointsType: str = None
    modelLink: str = None
    hash: str = None
    dimension: str = None

    def mandatoryCheck(self) -> bool:
        if not super().mandatoryCheck():
            return False
        
        if not _check(self.mode):
            return False
        
        if self.mode is FeatureBasedVideoMode.KeypointsDescriptionMode:
            if not _check(self.resolution):
                return False
            if not _check(self.framerate):
                return False
            if not _check(self.keypointsType):
                return False
            
        elif self.mode is FeatureBasedVideoMode.SharedNeuralNetworkMode:
            if not _check(self.modelLink):
                return False
            if not _check(self.hash):
                return False
            if not _check(self.dimension):
                return False
        
        return True

@dataclass
class ChannelAudio(Channel):
    """
    Audio를 위한 채널
    """
    def __init__(
        self,
        channelId: str = None,
        codec: AudioCodec = None,
        sampleRate: AudioSampleRate = None,
        bitrate: AudioBitrate = None,
        mono: AudioMono = None
    ):
        self.channelId = channelId
        self.channelType = ChannelType.Audio
        self.codec = codec
        self.sampleRate = sampleRate
        self.bitrate = bitrate
        self.mono = mono

    codec: AudioCodec = None
    sampleRate: AudioSampleRate = None
    bitrate: AudioBitrate = None
    mono: AudioMono = None

    def mandatoryCheck(self) -> bool:
        if not super().mandatoryCheck():
            return False
        
        if not _check(self.codec):
            return False
        if not _check(self.sampleRate):
            return False
        if not _check(self.bitrate):
            return False
        if not _check(self.mono):
            return False
        
        return True

@dataclass
class ChannelText(Channel):
    """
    text를 위한 채널
    """
    def __init__(self, channelId: str = None, format: TextFormat = None):
        self.channelId = channelId
        self.channelType = ChannelType.Text
        self.format = format

    format: TextFormat = None
    
    def mandatoryCheck(self) -> bool:
        if not super().mandatoryCheck():
            return False
        
        if not _check(self.format):
            return False
        
        return True

@dataclass
class CreationRequest(Request):
    """
    신규 서비스 생성 요청 Class
        mandatory:
            title, ownerId, adminKey, channelList
    """
    title: str = None
    description: str = None
    ownerId: str = None
    adminKey: str = None
    accessKey: str = None
    peerList: List[str] = None
    blockList: List[str] = None
    startDateTime: str = None #YYYYMMDDHHmmSS
    endDataTime: str = None #YYYYMMDDHHmmSS
    sourceList: List[str] = None
    channelList: List[Channel] = None

    def mandatoryCheck(self) -> bool:
        if not _check(self.title):
            return False
        if not _check(self.ownerId):
            return False
        if not _check(self.adminKey):
            return False
        if not _check(self.channelList):
            return False
        
        return True

@dataclass
class CreationResponse(Response):
    overlayId: str = None

@dataclass
class ModificationRequest(Request):
    """
    서비스 세션 변경 요청 Class
        mandatory:
            overlayId, ownerId, adminKey
            3개의 mandatory 는 변경할 서비스 세션을 식별하기 위한 키값으로 
            기존 서비스 세션의 값을 그대로 넣어야 하며 변경이 불가하다.

        optional:
            나머지 다른 값들은 변경이 필요한 값만 변경할 값으로 넣는다.

            ** channelList:
                channel의 channelType은 변경할 수 없으며 변경할 channel의 키값이다.
                sourceList만 변경 가능하다.
    """
    overlayId: str = None
    ownerId: str = None
    adminKey: str = None
    title: str = None
    description: str = None
    newOwnerId: str = None
    newAdminKey: str = None
    startDateTime: str = None #YYYYMMDDHHmmSS
    endDateTime: str = None #YYYYMMDDHHmmSS
    accessKey: str = None
    peerList: List[str] = None
    blockList: List[str] = None
    sourceList: List[str] = None
    channelList: List[Channel] = None

    def mandatoryCheck(self) -> bool:
        if not _check(self.overlayId):
            return False
        if not _check(self.ownerId):
            return False
        if not _check(self.adminKey):
            return False
        
        return True

@dataclass
class JoinRequest(Request):
    """
    서비스 세션 참가 요청 Class
        mandatory:
            overlayId, peerId, displayName, publicKeyPath, privateKeyPath
            
        optional:
            참가하려는 서비스 세션에 accessKey가 설정된 경우 accessKey 필수
    """
    overlayId: str = None
    accessKey: str = None
    peerId: str = None
    displayName: str = None
    publicKeyPath: str = None
    privateKeyPath: str = None
    
    def mandatoryCheck(self) -> bool:
        if not _check(self.overlayId):
            return False
        if not _check(self.peerId):
            return False
        if not _check(self.displayName):
            return False
        if not _check(self.publicKeyPath):
            return False
        if not _check(self.privateKeyPath):
            return False
        
        return True

@dataclass
class JoinResponse(Response):
    startDateTime: str = None #YYYYMMDDHHmmSS
    endDateTime: str = None #YYYYMMDDHHmmSS
    sourceList: List[str] = None
    channelList: List[Channel] = None

@dataclass
class Peer:
    """
    Peer 정보 Class
    """
    peerId: str = None
    displayName: str = None

@dataclass
class SearchPeerRequest(Request):
    """
    서비스 세션에 참가한 Peer 정보 요청 Class
        mandatory:
            overlayId
    """
    overlayId: str = None
        
    def mandatoryCheck(self) -> bool:
        if not _check(self.overlayId):
            return False
        
        return True

@dataclass
class SearchPeerResponse(Response):
    peerList: List[Peer] = None

@dataclass
class SendDataRequest(Request):
    """
    Data broadcast 요청 Class
        mandatory:
            dataType, overlayId, dataLength, data
    """
    dataType: DataType = None
    overlayId: str = None
    data: bytes = None
        
    def mandatoryCheck(self) -> bool:
        if not _check(self.dataType):
            return False
        if not _check(self.overlayId):
            return False
        if not _check(self.data):
            return False
        
        return True

@dataclass
class LeaveRequest(Request):
    """
    서비스 세션 탈퇴 요청 Class
        mandatory:
            overlayId, peerId
        optional:
            accessKey - 설정된 경우 필수
    """
    overlayId: str = None
    peerId: str = None
    accessKey: str = None
        
    def mandatoryCheck(self) -> bool:
        if not _check(self.overlayId):
            return False
        if not _check(self.peerId):
            return False
        
        return True

@dataclass
class RemovalRequest(Request):
    """
    서비스 세션 삭제 요청 Class
        mandatory:
            overlayId, ownerId, adminKey
    """
    overlayId: str = None
    ownerId: str = None
    adminKey: str = None
        
    def mandatoryCheck(self) -> bool:
        if not _check(self.overlayId):
            return False
        if not _check(self.ownerId):
            return False
        if not _check(self.adminKey):
            return False
        
        return True

@dataclass
class Notification:
    notificationType: NotificationType = None
    overlayId: str = None

@dataclass
class SessionChangeNotification(Notification):
    """
    세션 변경 알림 Class
    """
    def __init__(
        self,
        overlayId: str = None,
        title: str = None,
        description: str = None,
        startDateTime: str = None, #YYYYMMDDHHmmSS
        endDateTime: str = None, #YYYYMMDDHHmmSS
        ownerId: str = None,
        accessKey: str = None,
        sourceList: List[str] = None,
        channelList: List[Channel] = None
    ):
        self.notificationType = NotificationType.SessionChangeNotification
        self.overlayId = overlayId
        self.title = title
        self.description = description
        self.startDateTime = startDateTime
        self.endDateTime = endDateTime
        self.ownerId = ownerId
        self.accessKey = accessKey
        self.sourceList = sourceList
        self.channelList = channelList

    title: str = None
    description: str = None
    startDateTime: str = None #YYYYMMDDHHmmSS
    endDateTime: str = None #YYYYMMDDHHmmSS
    ownerId: str = None
    accessKey: str = None
    sourceList: List[str] = None
    channelList: List[Channel] = None

@dataclass
class SessionTerminationNotification(Notification):
    """
    세션 종료 알림 Class
    """
    def __init__(self, overlayId: str = None):
        self.notificationType = NotificationType.SessionTerminationNotification
        self.overlayId = overlayId

@dataclass
class PeerChangeNotification(Notification):
    """
    참가, 탈퇴하는 Peer 알림 Class
    """
    def __init__(
        self,
        overlayId: str = None,
        peerId: str = None,
        displayName: str = None,
        leave: bool = None
    ):
        self.notificationType = NotificationType.PeerChangeNotification
        self.overlayId = overlayId
        self.peerId = peerId
        self.displayName = displayName
        self.leave = leave
    
    peerId: str = None
    displayName: str = None
    leave: bool = None

@dataclass
class DataNotification(Notification):
    """
    Broadcast data 수신 Class
    """
    def __init__(
        self,
        overlayId: str = None,
        dataType: DataType = None,
        peerId: str = None,
        data: bytes = None
    ):
        self.notificationType = NotificationType.DataNotification
        self.overlayId = overlayId
        self.dataType = dataType
        self.peerId = peerId
        self.data = data

    dataType: DataType = None
    peerId: str = None
    data: bytes = None