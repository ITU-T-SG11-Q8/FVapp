import torch
import numpy as np
import cv2
import time
import struct
from dataclasses import dataclass
import zlib

PROTOCOL_VERSION = 1
SSRC_TYPE = 12341234

MEDIA_TYPE_VIDEO = 1000
MEDIA_TYPE_AUDIO = 2000
MEDIA_TYPE_DATA = 3000

KEY_FRAME_TYPE = 1100
KEY_REQ_FRAME_TYPE = 1101

AVATARIFY_TYPE = 1200
AVATARIFY_VALUE_TYPE = 1201
AVATARIFY_JACOBIAN_TYPE = 1201

COMPRESS_ZIP = 1

@dataclass
class TlvData:
    #tlvType: int = 0
    #tlvLength: int = 0
    #tlvValue: any = None

    def from_tlv(self, data):
        _read_pos = 0

        _type = data[_read_pos:_read_pos + 2]
        _type = np.frombuffer(_type, dtype=np.int32)[0]
        _read_pos += 2

        _length = data[_read_pos:_read_pos + 4]
        _length = np.frombuffer(_length, dtype=np.int32)[0]
        _read_pos += 4

        _value = data[_read_pos:_read_pos + _length]
        _read_pos += _length

        return _type, _length, _value, data[_read_pos:]

    def to_tlv(self, _type, bin_data):
        bin_type = np.array([_type], dtype=np.int32)
        bin_type = np.frombuffer(bin_type.tobytes(), dtype=np.uint8)

        if type(bin_data) is not bytes:
            bin_data = bin_data.tobytes()
        bin_data = np.frombuffer(bin_data, dtype=np.uint8)

        bin_len = np.array([len(bin_data)], dtype=np.int32)
        bin_len = np.frombuffer(bin_len.tobytes(), dtype=np.uint8)

        tlv_data = np.concatenate([bin_type, bin_len, bin_data])
        return tlv_data


@dataclass
class MediaData:
    compressFormat: int = 0,
    compressDataLength: int = 0,
    #data: TlvData = TlvData(),
    mediaDataType: int = 0,
    kp_norm: any = None,
    device: any = None

    def parse_kp_norm(self, bin_data, device):
        value = None
        jacobian = None
        tlv_data: TlvData = TlvData()

        while len(bin_data) > 0:
            _type, length, _value, bin_data = tlv_data.from_tlv(bin_data)
            _value = np.frombuffer(_value, dtype=np.float32)

            if _type == AVATARIFY_VALUE_TYPE:
                _value = _value.reshape(1, 10, 2)
                value = torch.as_tensor(_value).to(device)
            elif _type == AVATARIFY_JACOBIAN_TYPE:
                _value = _value.reshape(1, 10, 2, 2)
                jacobian = torch.as_tensor(_value).to(device)

        torch_data = {'value': value, 'jacobian': jacobian}
        return torch_data

    def to_bin_kp_norm(self):
        to_bin = bytes()
        tlv_data: TlvData = TlvData()

        dic_name = ('value', 'jacobian')
        dic_type = (AVATARIFY_VALUE_TYPE, AVATARIFY_JACOBIAN_TYPE)
        for i in range(len(dic_name)):
            _type = dic_type[i]

            name = dic_name[i]
            value = self.kp_norm[name]
            value = value.cpu().numpy()
            value = value.reshape(-1, )

            bin_data = tlv_data.to_tlv(_type, value)
            to_bin = to_bin + bin_data.tobytes()

        to_bin = tlv_data.to_tlv(AVATARIFY_TYPE, to_bin)

        return to_bin.tobytes()

    def parse_data(self, media_type, data):
        _read_pos = 0
        _compress_type = data[_read_pos:_read_pos + 2]
        self.compressFormat = np.frombuffer(_compress_type, dtype=np.int16)[0]
        _read_pos += 2

        _media_data_len = data[_read_pos:_read_pos + 4]
        self.compressDataLength = np.frombuffer(_media_data_len, dtype=np.int16)[0]
        _read_pos += 4

        _data = data[_read_pos:_read_pos + _media_data_len]

        if media_type == AVATARIFY_TYPE:
            self.kp_norm = self.parse_kp_norm(_data, self.device)
        else:
            self.data.type, self.data.length, self.data.value, rest_bin_data = self.data.from_tlv(_data)

    def to_data(self, media_type, raw_data):
        _compress_type = np.array([self.compressFormat], dtype=np.int16)
        _compress_type = np.frombuffer(_compress_type.tobytes(), dtype=np.uint8)

        _bin_data = None
        #print(f"media_type:{media_type}")
        if media_type == AVATARIFY_TYPE:
            _bin_data = self.to_bin_kp_norm()
        else:
            tlv_data: TlvData = TlvData()
            _bin_data = tlv_data.to_tlv(media_type, raw_data)

        _media_len = np.array([len(_bin_data)], dtype=np.int32)
        _media_len = np.frombuffer(_media_len.tobytes(), dtype=np.uint8)

        _data = np.frombuffer(_bin_data, dtype=np.uint8)

        data = np.concatenate([_compress_type, _media_len, _data])
        return data


@dataclass
class CommonData:
    version: int = 0,
    timeStamp: float = 0.0,
    sequenceNumber: int = 0,
    ssrc: int = 0,
    mediaType: int = 0,
    mediaLength: int = 0,
    mediaData: MediaData = MediaData()

    def parse_data(self, data):
        _read_pos = 0
        _version = data[_read_pos:_read_pos + 2]
        self.version = np.frombuffer(_version, dtype=np.int16)[0]
        _read_pos += 2

        _timestamp = data[_read_pos:_read_pos + 8]
        self.timeStamp = np.frombuffer(_timestamp, dtype=np.float32)[0]
        _read_pos += 8

        _seq_number = data[_read_pos:_read_pos + 4]
        self.sequenceNumber = np.frombuffer(_seq_number, dtype=np.int32)[0]
        _read_pos += 4

        _ssrc = data[_read_pos:_read_pos + 4]
        self.ssrc = np.frombuffer(_ssrc, dtype=np.int32)[0]
        _read_pos += 4

        _type = data[_read_pos:_read_pos + 2]
        self.mediaType = np.frombuffer(_type, dtype=np.int16)[0]
        _read_pos += 2

        _length = data[_read_pos:_read_pos + 4]
        self.mediaLength = np.frombuffer(_length, dtype=np.int32)[0]
        _read_pos += 4

        _data = data[_read_pos:_read_pos + _length]
        _read_pos += _length

        if self.mediaLength > 0:
            self.mediaData.parse_data(_data)

    def to_data(self, raw_data):
        _version = np.array([PROTOCOL_VERSION], dtype=np.int16)
        _version = np.frombuffer(_version.tobytes(), dtype=np.uint8)

        timestamp = time.time()
        _timestamp = np.array([timestamp], dtype=np.float32)
        _timestamp = np.frombuffer(_timestamp.tobytes(), dtype=np.uint8)

        _seq_number = np.array([self.sequenceNumber], dtype=np.int32)
        _seq_number = np.frombuffer(_seq_number.tobytes(), dtype=np.uint8)

        _ssrc = np.array([SSRC_TYPE], dtype=np.int32)
        _ssrc = np.frombuffer(_ssrc.tobytes(), dtype=np.uint8)

        _common_data_type = np.array([self.mediaType], dtype=np.int16)
        _common_data_type = np.frombuffer(_common_data_type.tobytes(), dtype=np.uint8)

        _bin_data = self.mediaData.to_data(self.mediaData.mediaDataType, raw_data)

        _common_data_len = np.array([len(_bin_data)], dtype=np.int32)
        _common_data_len = np.frombuffer(_common_data_len.tobytes(), dtype=np.uint8)

        if type(_bin_data) is not bytes:
            data = _bin_data.tobytes()
        data = np.frombuffer(_bin_data, dtype=np.uint8)

        _data = np.concatenate([_version, _timestamp, _seq_number, _ssrc, _common_data_type, _common_data_len, data])
        return _data

    def is_keyframe_type(self):
        if self.mediaType == KEY_FRAME_TYPE:
            return True
        return False

    def is_avatarify_type(self):
        if self.mediaType == AVATARIFY_TYPE:
            return True
        return False

@dataclass
class PacketData:
    type: int = 0,
    length: int = 0,
    common_data: CommonData = None

    def __init__(self, tlv_type):
        self.type = tlv_type
        self.common_data = CommonData()

    def parse_data(self, bin_data):
        _read_pos = 0

        _type = bin_data[_read_pos:_read_pos + 2]
        self.type = np.frombuffer(_type, dtype=np.int32)[0]
        _read_pos += 2

        _length = bin_data[_read_pos:_read_pos + 4]
        self.length = np.frombuffer(_length, dtype=np.int32)[0]
        _read_pos += 4

        _data = bin_data[_read_pos:_read_pos + _length]
        _read_pos += _length

        if self.length > 0:
            self.common_data.parse_data(_data)

    def to_data(self, raw_data):
        bin_type = np.array([self.type], dtype=np.int16)
        bin_type = np.frombuffer(bin_type.tobytes(), dtype=np.uint8)

        bin_data = self.common_data.to_data(raw_data)

        bin_len = np.array([len(bin_data)], dtype=np.int32)
        bin_len = np.frombuffer(bin_len.tobytes(), dtype=np.uint8)

        data = np.concatenate([bin_type, bin_len, bin_data])
        return data


class BINWrapper:
    sequence_number: int = 1

    def get_sequence_number(self):
        self.sequence_number += 1
        if self.sequence_number < 0:
            self.sequence_number = 0
        return self.sequence_number

    def make_send_keyframe_packet(self, frame, compress_format):
        _sendData: PacketData = PacketData(MEDIA_TYPE_VIDEO)

        _sendData.common_data.mediaData.mediaDataType = KEY_FRAME_TYPE
        #_sendData.common_data.mediaData.data.length = len(frame)
        #_sendData.common_data.mediaData.data.value = frame

        _sendData.common_data.mediaData.compressFormat = compress_format

        _sendData.common_data.version = PROTOCOL_VERSION
        _sendData.common_data.timeStamp = time.time()
        _sendData.common_data.sequenceNumber = self.get_sequence_number()
        _sendData.common_data.ssrc = SSRC_TYPE
        _sendData.common_data.mediaType = MEDIA_TYPE_VIDEO

        data = _sendData.to_data(frame)
        return data

    def make_send_avatarify_packet(self, kp_norm, compress_format):
        _sendData: PacketData = PacketData(MEDIA_TYPE_VIDEO)

        _sendData.common_data.mediaData.mediaDataType = AVATARIFY_TYPE
        _sendData.common_data.mediaData.kp_norm = kp_norm

        _sendData.common_data.mediaData.compressFormat = compress_format

        _sendData.common_data.version = PROTOCOL_VERSION
        _sendData.common_data.timeStamp = time.time()
        _sendData.common_data.sequenceNumber = self.get_sequence_number()
        _sendData.common_data.ssrc = SSRC_TYPE
        _sendData.common_data.mediaType = MEDIA_TYPE_VIDEO

        data = _sendData.to_data(kp_norm)
        return data

    def make_send_audio_packet(self, value, compress_format):
        _sendData: PacketData = PacketData(MEDIA_TYPE_AUDIO)

        #_sendData.common_data.mediaData.data.type = MEDIA_TYPE_AUDIO
        #_sendData.common_data.mediaData.data.value = value

        _sendData.common_data.mediaData.compressFormat = compress_format

        _sendData.common_data.version = PROTOCOL_VERSION
        _sendData.common_data.timeStamp = time.time()
        _sendData.common_data.sequenceNumber = self.get_sequence_number()
        _sendData.common_data.ssrc = SSRC_TYPE
        _sendData.common_data.mediaType = MEDIA_TYPE_AUDIO

        data = _sendData.to_data(value)
        return data

    def make_send_data_packet(self, value, compress_format):
        _sendData: PacketData = PacketData(MEDIA_TYPE_DATA)

        #_sendData.common_data.mediaData.data.type = MEDIA_TYPE_DATA
        #_sendData.common_data.mediaData.data.value = value

        _sendData.common_data.mediaData.compressFormat = compress_format

        _sendData.common_data.version = PROTOCOL_VERSION
        _sendData.common_data.timeStamp = time.time()
        _sendData.common_data.sequenceNumber = self.get_sequence_number()
        _sendData.common_data.ssrc = SSRC_TYPE
        _sendData.common_data.mediaType = MEDIA_TYPE_DATA

        data = _sendData.to_data()
        return data

    def parser_packet(self, recv_bin_data):
        packet: PacketData = PacketData()
        packet.parse_data(recv_bin_data)
        return packet

    def to_data(self, send_data: PacketData):
        bin_data = send_data.to_data()
        return bin_data

    def to_tlv(self, _type, bin_data):
        bin_type = np.array([_type], dtype=np.int16)
        bin_type = np.frombuffer(bin_type.tobytes(), dtype=np.uint8)

        if type(bin_data) is not bytes:
            bin_data = bin_data.tobytes()
        bin_data = np.frombuffer(bin_data, dtype=np.uint8)

        bin_len = np.array([len(bin_data)], dtype=np.int32)
        bin_len = np.frombuffer(bin_len.tobytes(), dtype=np.uint8)

        tlv_data = np.concatenate([bin_type, bin_len, bin_data])
        return tlv_data

    def from_tlv(self, bin_data):
        _read_pos = 0

        _type = bin_data[_read_pos:_read_pos + 4]
        _type = np.frombuffer(_type, dtype=np.int32)[0]
        _read_pos += 4

        _length = bin_data[_read_pos:_read_pos + 4]
        _length = np.frombuffer(_length, dtype=np.int32)[0]
        _read_pos += 4

        _value = bin_data[_read_pos:_read_pos + _length]
        _read_pos += _length

        return _type, _length, _value, bin_data[_read_pos:]

    def to_bin_key_frame(self, frame):
        _type = 100
        to_bin = self.to_tlv(_type, frame)

        return to_bin.tobytes()

    def parse_bin(self, bin_data):
        _type, length, value, bin_data = self.from_tlv(bin_data)
        return _type, value, bin_data

    def parse_key_frame(self, frame):
        # print(f'### {frame}')
        frame = np.frombuffer(frame, dtype=np.uint8)
        frame = cv2.imdecode(frame, flags=1)
        return frame




