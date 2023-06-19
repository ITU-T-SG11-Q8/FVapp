import torch
import numpy as np
import cv2
import struct

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

    def to_bin_kp_norm(self, kp_norm):
        to_bin = bytes()

        dic_name = ('value', 'jacobian')
        dic_type = (201, 202)
        for i in range(len(dic_name)):
            _type = dic_type[i]

            name = dic_name[i]
            value = kp_norm[name]
            value = value.cpu().numpy()
            value = value.reshape(-1, )

            bin_data = self.to_tlv(_type, value)
            #print(f'### type:{_type} size:{len(bin_data.tobytes())}')
            to_bin = to_bin + bin_data.tobytes()

        _type = 200
        to_bin = self.to_tlv(_type, to_bin)

        #print(f'### type:{_type} size:{len(to_bin.tobytes())}')
        return to_bin.tobytes()

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

    def parse_kp_norm(self, bin_data, device):
        value = None
        jacobian = None

        while len(bin_data) > 0:
            _type, length, _value, bin_data = self.from_tlv(bin_data)
            _value = np.frombuffer(_value, dtype=np.float32)

            if _type == 201:
                _value = _value.reshape(1, 10, 2)
                value = torch.as_tensor(_value).to(device)
            elif _type == 202:
                _value = _value.reshape(1, 10, 2, 2)
                jacobian = torch.as_tensor(_value).to(device)

        torch_data = {'value': value, 'jacobian': jacobian}
        return torch_data



