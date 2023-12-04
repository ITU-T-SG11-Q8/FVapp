import torch
import numpy as np
import cv2
import struct

class SPIGAPacket:
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

    def to_bin_features(self, frame, features_tracker, features_spiga):
        to_bin = bytes()

        shape = np.array(frame.shape)
        bin_data = self.to_tlv(1301, shape)
        to_bin = to_bin + bin_data.tobytes()

        features_name = ('tracker', 'spiga')
        features_values = (features_tracker, features_spiga)
        dic_names = [('bbox', 'landmarks'), ('landmarks', 'headpose')]
        dic_types = [(1302, 1303),(1304, 1305)]

        for i in range(len(features_name)):
            features_value = features_values[i]
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

        _type = 1300
        to_bin = self.to_tlv(_type, to_bin)

        #print(f'### type:{_type} size:{len(to_bin.tobytes())}')
        return to_bin.tobytes()

    def parse_bin(self, bin_data):
        _type, length, value, bin_data = self.from_tlv(bin_data)
        return _type, value, bin_data

    def parse_features(self, bin_data):
        shape = None
        tracker_bbox = None
        tracker_landmarks = None
        spiga_landmarks = None
        spiga_headpose = None

        while len(bin_data):
            inner_type, inner_length, inner_value, bin_data = self.from_tlv(bin_data)

            if inner_type == 1301:
                shape = np.frombuffer(inner_value, dtype=int)
            elif inner_type == 1302:
                inner_value = np.frombuffer(inner_value, dtype=np.float32)
                #tracker_bbox = inner_value.reshape(1, 5)
                tracker_bbox = inner_value.reshape(1, -1)
            elif inner_type == 1303:
                inner_value = np.frombuffer(inner_value, dtype=np.float32)
                #tracker_landmarks = inner_value.reshape(1, 5, 2)
                tracker_landmarks = inner_value.reshape(1, -1, 2)
            elif inner_type == 1304:
                inner_value = np.frombuffer(inner_value, dtype=float)
                #spiga_landmarks = inner_value.reshape(1, 98, 2)
                spiga_landmarks = inner_value.reshape(1, -1, 2)
                spiga_landmarks = list(spiga_landmarks)
            elif inner_type == 1305:
                inner_value = np.frombuffer(inner_value, dtype=float)
                #spiga_headpose = inner_value.reshape(1, 6)
                spiga_headpose = inner_value.reshape(1, -1)
                spiga_headpose = list(spiga_headpose)

        features_tracker = {"bbox": tracker_bbox, "landmarks": tracker_landmarks}
        features_spiga = {"landmarks": spiga_landmarks, "headpose": spiga_headpose}

        return shape, features_tracker, features_spiga



