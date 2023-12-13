import torch
import numpy as np
import cv2
import struct

class TYPE_INDEX:
    TYPE_VIDEO_KEY_FRAME = 1000
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

class BINWrapper:
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
        _read_pos += 4

        _length = bin_data[_read_pos:_read_pos + 4]
        _length = np.frombuffer(_length, dtype=np.int32)[0]
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
        return to_bin.tobytes()

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
        return to_bin.tobytes()

    def to_bin_key_frame(self, frame):
        _type = TYPE_INDEX.TYPE_VIDEO_KEY_FRAME
        to_bin = self.to_tlv(_type, frame)

        return to_bin.tobytes()

    def to_bin_audio_data(self, frame):
        _type = TYPE_INDEX.TYPE_AUDIO
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

    def parse_kp_norm(self, bin_data, device):
        value = None
        jacobian = None

        while len(bin_data) > 0:
            _type, length, _value, bin_data = self.from_tlv(bin_data)
            _value = np.frombuffer(_value, dtype=np.float32)

            if _type == TYPE_INDEX.TYPE_VIDEO_AVATARIFY_KP_NORM:
                _value = _value.reshape(1, 10, 2)
                value = torch.as_tensor(_value).to(device)
            elif _type == TYPE_INDEX.TYPE_VIDEO_AVATARIFY_JACOBIAN:
                _value = _value.reshape(1, 10, 2, 2)
                jacobian = torch.as_tensor(_value).to(device)

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



