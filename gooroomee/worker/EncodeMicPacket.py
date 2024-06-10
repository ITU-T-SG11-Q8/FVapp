import pyaudio
from PyQt5.QtWidgets import QApplication
import time

from gooroomee.grm_defs import GrmParentThread, MIC_CHUNK, SPK_CHUNK
from gooroomee.grm_packet import BINWrapper, TYPE_INDEX
from gooroomee.grm_queue import GRMQueue


def get_current_time_ms():
    return round(time.time() * 1000)


class EncodeMicPacketWorker(GrmParentThread):
    def __init__(self,
                 p_send_grm_queue,
                 p_get_worker_seq_num,
                 p_get_worker_ssrc):
        super().__init__()
        self.mic_stream = None
        self.polled_count = 0
        self.work_done = 0
        self.receive_data = 0
        self.send_grm_queue: GRMQueue = p_send_grm_queue
        self.get_worker_seq_num = p_get_worker_seq_num
        self.get_worker_ssrc = p_get_worker_ssrc
        self.mic_interface = 0
        self.connect_flag: bool = False
        self.bin_wrapper = BINWrapper()
        self.device_index: int = 0
        self.enable_mic: bool = False
        self.failed_mic: bool = False
        self.changing_device: bool = False

    def set_join(self, p_join_flag: bool):
        GrmParentThread.set_join(self, p_join_flag)
        self.send_grm_queue.clear()

        if p_join_flag is False:
            self.set_enable_mic(False)

    def set_connect(self, p_connect_flag: bool):
        self.connect_flag = p_connect_flag
        print(f"EncodeMicPacketWorker connect:{self.connect_flag}")

    def set_enable_mic(self, enable_mic):
        self.enable_mic = enable_mic
        print(f"EncodeMicPacketWorker enable_mic:{self.enable_mic}")

    def change_device_mic(self, p_device_index):
        self.failed_mic = False
        self.changing_device = True

        print(f'try to change mic device index = [{p_device_index}]')
        while self.mic_interface is not None:
            time.sleep(0.1)

        self.device_index = p_device_index
        print(f'completed changing mic device index = [{p_device_index}]')
        self.changing_device = False

    def run(self):
        while self.alive:
            while self.running:
                if self.join_flag is False or \
                        self.enable_mic is False or \
                        self.failed_mic is True or \
                        self.changing_device is True:
                    time.sleep(0.001)
                    continue

                try:
                    self.mic_interface = pyaudio.PyAudio()
                    print(f"Mic Open, Mic Index:{self.device_index}")
                    self.mic_stream = self.mic_interface.open(format=pyaudio.paInt16,
                                                              channels=1,
                                                              rate=44100,
                                                              input=True,
                                                              input_device_index=self.device_index,
                                                              frames_per_buffer=MIC_CHUNK)
                except Exception as e:
                    print(f'{e}')

                if self.mic_stream is None:
                    self.failed_mic = True
                    time.sleep(0.001)
                    break

                print(f"Mic End, Mic Index:{self.device_index} mic_stream:{self.mic_stream}")

                while self.running:
                    if self.join_flag is False or \
                            self.enable_mic is False or \
                            self.changing_device is True:
                        break

                    _frames = self.mic_stream.read(SPK_CHUNK, exception_on_overflow=False)
                    if _frames is None:
                        time.sleep(0.001)
                        continue

                    audio_bin_data = self.bin_wrapper.to_bin_audio_data(_frames)
                    audio_bin_data = self.bin_wrapper.to_bin_wrap_common_header(timestamp=get_current_time_ms(),
                                                                                seq_num=self.get_worker_seq_num(),
                                                                                ssrc=self.get_worker_ssrc(),
                                                                                mediatype=TYPE_INDEX.TYPE_AUDIO,
                                                                                bindata=audio_bin_data)
                    self.send_grm_queue.put(audio_bin_data)
                    time.sleep(0.001)

                self.mic_stream.stop_stream()
                self.mic_stream.close()
                self.mic_interface.terminate()
                self.mic_stream = None
                self.mic_interface = None

                QApplication.processEvents()
                time.sleep(0.001)

            QApplication.processEvents()
            time.sleep(0.001)

        print("Stop EncodeMicPacketWorker")
        self.terminated = True
        # self.terminate()
