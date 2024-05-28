import torch
import numpy as np
import cv2
import struct

class TYPE_INDEX:
    TYPE_VIDEO = 1000
    TYPE_VIDEO_KEY_FRAME = 1100
    TYPE_VIDEO_KEY_FRAME_REQUEST = 1101
    TYPE_VIDEO_AVATARIFY = 1200
    TYPE_VIDEO_AVATARIFY_KP_NORM = 1201
    TYPE_VIDEO_AVATARIFY_JACOBIAN = 1202
    TYPE_VIDEO_SPIGA = 1300
    TYPE_VIDEO_SPIGA_SHAPE = 1301
    TYPE_VIDEO_SPIGA_TRACKER_BBOX = 1302
    TYPE_VIDEO_SPIGA_TRACKER_LANDMARKS = 1303
    TYPE_VIDEO_SPIGA_SPIGA_LANDMARKS = 1304
    TYPE_VIDEO_SPIGA_SPIGA_HEADPOSE = 1305
    TYPE_AUDIO = 2000
    TYPE_AUDIO_ZIP = 2100
    TYPE_DATA = 3000
    TYPE_DATA_CHAT = 3100

class BINWrapper:
    def array_to_int(self, in_array):
        x = 0
        for c in in_array:
            x <<= 8
            # x |= c
            x = int(np.uint64(x + c))
        return x

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

    def from_tlv(self, bin_data):
        _read_pos = 0

        _type = bin_data[_read_pos:_read_pos + 4]
        _type = np.frombuffer(_type, dtype=np.int32)[0]
        #_type = self.array_to_int(_type)
        _read_pos += 4

        _length = bin_data[_read_pos:_read_pos + 4]
        _length = np.frombuffer(_length, dtype=np.int32)[0]
        #_length = self.array_to_int(_length)
        _read_pos += 4

        _value = bin_data[_read_pos:_read_pos + _length]
        _read_pos += _length

        return _type, _length, _value, bin_data[_read_pos:]

    #def to_bin_video(self, param):
    #    if self.comm_mode_type == True:
    #        return self._to_bin_kp_norm(param)
    #    else:
    #        return self._to_bin_features(param)

    def to_bin_kp_norm(self, kp_norm):
        to_bin = bytes()

        dic_name = ('value', 'jacobian')
        dic_type = (TYPE_INDEX.TYPE_VIDEO_AVATARIFY_KP_NORM, TYPE_INDEX.TYPE_VIDEO_AVATARIFY_JACOBIAN)
        for i in range(len(dic_name)):
            _type = dic_type[i]

            name = dic_name[i]
            value = kp_norm[name]
            value = value.cpu().numpy()
            value = value.reshape(-1, )

            bin_data = self.to_tlv(_type, value)
            #print(f'### type:{_type} size:{len(bin_data.tobytes())}')
            to_bin = to_bin + bin_data.tobytes()

        _type = TYPE_INDEX.TYPE_VIDEO_AVATARIFY
        to_bin = self.to_tlv(_type, to_bin)

        #print(f'### type:{_type} size:{len(to_bin.tobytes())}')
        return to_bin       #.tobytes()

    def to_bin_features(self, frame, features_tracker, features_spiga):
        to_bin = bytes()

        shape = np.array(frame.shape)
        bin_data = self.to_tlv(TYPE_INDEX.TYPE_VIDEO_SPIGA_SHAPE, shape)
        to_bin = to_bin + bin_data.tobytes()

        features_name = ('tracker', 'spiga')
        features_values = (features_tracker, features_spiga)
        dic_names = [('bbox', 'landmarks'), ('landmarks', 'headpose')]
        dic_types = [(TYPE_INDEX.TYPE_VIDEO_SPIGA_TRACKER_BBOX, TYPE_INDEX.TYPE_VIDEO_SPIGA_TRACKER_LANDMARKS), (TYPE_INDEX.TYPE_VIDEO_SPIGA_SPIGA_LANDMARKS, TYPE_INDEX.TYPE_VIDEO_SPIGA_SPIGA_HEADPOSE)]

        for i in range(len(features_name)):
            features_value = features_values[i]
            if features_value is None:
                return None

            dic_name = dic_names[i]
            dic_type = dic_types[i]
            for j in range(len(dic_name)):
                _type = dic_type[j]

                name = dic_name[j]
                value = features_value[name]
                if i > 0:
                    value = np.array(value)
                value = value.reshape(-1, )

                bin_data = self.to_tlv(_type, value)
                to_bin = to_bin + bin_data.tobytes()

        _type = TYPE_INDEX.TYPE_VIDEO_SPIGA
        to_bin = self.to_tlv(_type, to_bin)

        # print(f'### type:{_type} size:{len(to_bin.tobytes())}')
        return to_bin       #.tobytes()

    def to_bin_key_frame(self, frame):
        _type = TYPE_INDEX.TYPE_VIDEO_KEY_FRAME
        to_bin = self.to_tlv(_type, frame)
        return to_bin       #.tobytes()

    def to_bin_request_key_frame(self):
        _type = TYPE_INDEX.TYPE_VIDEO_KEY_FRAME_REQUEST
        to_bin = self.to_tlv(_type, b'')
        return to_bin       #.tobytes()

    def to_bin_audio_data(self, frame):
        _type = TYPE_INDEX.TYPE_AUDIO_ZIP
        to_bin = self.to_tlv(_type, frame)
        return to_bin       #.tobytes()

    def to_bin_chat_data(self, chat_data):
        chat_data = bytes(chat_data, 'utf-8')
        _type = TYPE_INDEX.TYPE_DATA_CHAT
        to_bin = self.to_tlv(_type, chat_data)
        return to_bin       #.tobytes()

    def to_bin_wrap_common_header(self, timestamp: np.uint64, seq_num: np.uint32, ssrc: np.uint32, mediatype: np.uint16, bindata, version: np.uint16 = 1):
        '''
        if bindata is None:
            bindata = b''
        bin_version = struct.pack('<H', version)            # H : unsigned short
        bin_timestamp = struct.pack('<Q', timestamp)        # Q : unsigned long long
        bin_seq_num = struct.pack('<L', seq_num)            # L : unsigned long
        bin_ssrc = struct.pack('<L', ssrc)                  # L : unsigned long
        bin_mediatype = struct.pack('<H', mediatype)        # H : unsigned short
        bin_bindata_len = struct.pack('<L', len(bindata))   # L : unsigned long
        '''

        bin_version = np.array([version], dtype=np.uint16)
        bin_version = np.frombuffer(bin_version.tobytes(), dtype=np.uint8)

        bin_timestamp = np.array([timestamp], dtype=np.uint64)
        bin_timestamp = np.frombuffer(bin_timestamp.tobytes(), dtype=np.uint8)

        bin_seq_num = np.array([seq_num], dtype=np.uint32)
        bin_seq_num = np.frombuffer(bin_seq_num.tobytes(), dtype=np.uint8)

        bin_ssrc = np.array([ssrc], dtype=np.uint32)
        bin_ssrc = np.frombuffer(bin_ssrc.tobytes(), dtype=np.uint8)

        bin_mediatype = np.array([mediatype], dtype=np.uint16)
        bin_mediatype = np.frombuffer(bin_mediatype.tobytes(), dtype=np.uint8)

        if bindata is None:
            bindata = []
        bin_bindata_len = np.array([len(bindata)], dtype=np.uint16)
        bin_bindata_len = np.frombuffer(bin_bindata_len.tobytes(), dtype=np.uint8)

        bin_data = np.concatenate([bin_version,
                                   bin_timestamp,
                                   bin_seq_num,
                                   bin_ssrc,
                                   bin_mediatype,
                                   bin_bindata_len,
                                   bindata])
        return bin_data.tobytes()

    def parse_bin(self, bin_data):
        _type, length, value, bin_data = self.from_tlv(bin_data)
        return _type, value, bin_data

    def parse_key_frame(self, frame):
        # print(f'### {frame}')
        frame = np.frombuffer(frame, dtype=np.uint8)
        frame = cv2.imdecode(frame, flags=1)
        return frame

    def parse_kp_norm(self, bin_data, device):
        value = None
        jacobian = None

        while len(bin_data) > 0:
            _type, length, _value, bin_data = self.from_tlv(bin_data)
            _value = np.frombuffer(_value, dtype=np.float32)

            try:
                if _type == TYPE_INDEX.TYPE_VIDEO_AVATARIFY_KP_NORM:
                    _value = _value.reshape(1, 10, 2)
                    value = torch.as_tensor(_value).to(device)
                elif _type == TYPE_INDEX.TYPE_VIDEO_AVATARIFY_JACOBIAN:
                    _value = _value.reshape(1, 10, 2, 2)
                    jacobian = torch.as_tensor(_value).to(device)
            except Exception as e:
                print(f'{e}')
                return None

        torch_data = {'value': value, 'jacobian': jacobian}
        return torch_data

    def parse_features(self, bin_data):
        shape = None
        tracker_bbox = None
        tracker_landmarks = None
        spiga_landmarks = None
        spiga_headpose = None

        while len(bin_data):
            inner_type, inner_length, inner_value, bin_data = self.from_tlv(bin_data)

            if inner_type == TYPE_INDEX.TYPE_VIDEO_SPIGA_SHAPE:
                shape = np.frombuffer(inner_value, dtype=int)
            elif inner_type == TYPE_INDEX.TYPE_VIDEO_SPIGA_TRACKER_BBOX:
                inner_value = np.frombuffer(inner_value, dtype=np.float32)
                tracker_bbox = inner_value.reshape(1, -1)           # 1, 5
            elif inner_type == TYPE_INDEX.TYPE_VIDEO_SPIGA_TRACKER_LANDMARKS:
                inner_value = np.frombuffer(inner_value, dtype=np.float32)
                tracker_landmarks = inner_value.reshape(1, -1, 2)   # 1, 5, 2
            elif inner_type == TYPE_INDEX.TYPE_VIDEO_SPIGA_SPIGA_LANDMARKS:
                inner_value = np.frombuffer(inner_value, dtype=float)
                spiga_landmarks = inner_value.reshape(1, -1, 2)     # 1, 98, 2
                spiga_landmarks = list(spiga_landmarks)
            elif inner_type == TYPE_INDEX.TYPE_VIDEO_SPIGA_SPIGA_HEADPOSE:
                inner_value = np.frombuffer(inner_value, dtype=float)
                spiga_headpose = inner_value.reshape(1, -1)         # 1, 6
                spiga_headpose = list(spiga_headpose)

        features_tracker = {"bbox": tracker_bbox, "landmarks": tracker_landmarks}
        features_spiga = {"landmarks": spiga_landmarks, "headpose": spiga_headpose}

        return shape, features_tracker, features_spiga

    def parse_chat(self, bin_data):
        bin_data = bytearray(bin_data)
        chat_message = np.array(bin_data, dtype=np.uint8)
        chat_message = str(chat_message, 'utf-8')

        return chat_message

    def parse_wrap_common_header(self, bin_data):
        _read_pos = 0

        _version = bin_data[_read_pos:_read_pos + 2]
        #_version = np.frombuffer(_version, dtype=np.uint16)[0]
        _version = self.array_to_int(_version[::-1])
        _read_pos += 2

        _timestamp = bin_data[_read_pos:_read_pos + 8]
        #_timestamp = np.frombuffer(_timestamp, dtype=np.uint64)[0]
        _timestamp = self.array_to_int(_timestamp[::-1])
        _read_pos += 8

        _seq_num = bin_data[_read_pos:_read_pos + 4]
        #_seq_num = np.frombuffer(_seq_num, dtype=np.uint32)[0]
        _seq_num = self.array_to_int(_seq_num[::-1])
        _read_pos += 4

        _ssrc = bin_data[_read_pos:_read_pos + 4]
        #_ssrc = np.frombuffer(_ssrc, dtype=np.uint32)[0]
        _ssrc = self.array_to_int(_ssrc[::-1])
        _read_pos += 4

        _mediatype = bin_data[_read_pos:_read_pos + 2]
        #_mediatype = np.frombuffer(_mediatype, dtype=np.uint16)[0]
        _mediatype = self.array_to_int(_mediatype[::-1])
        _read_pos += 2

        _bindata_len = bin_data[_read_pos:_read_pos + 2]
        #_bindata_len = np.frombuffer(_bindata_len, dtype=np.uint16)[0]
        _bindata_len = self.array_to_int(_bindata_len[::-1])
        _read_pos += 2

        _bindata = bin_data[_read_pos:_read_pos + _bindata_len]
        _read_pos += _bindata_len

        return _version, _timestamp, _seq_num, _ssrc, _mediatype, _bindata_len, _bindata

