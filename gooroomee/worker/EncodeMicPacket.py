import pyaudio
from PyQt5.QtWidgets import QApplication
import time

from gooroomee.grm_defs import GrmParentThread, MIC_CHUNK, SPK_CHUNK
from gooroomee.grm_packet import BINWrapper, TYPE_INDEX
from gooroomee.grm_queue import GRMQueue


class EncodeMicPacketWorker(GrmParentThread):
    def __init__(self,
                 p_send_grm_queue,
                 p_get_current_milli_time,
                 p_get_worker_seqnum,
                 p_get_worker_ssrc):
        super().__init__()
        self.mic_stream = None
        self.polled_count = 0
        self.work_done = 0
        self.receive_data = 0
        self.send_grm_queue: GRMQueue = p_send_grm_queue
        self.get_current_milli_time = p_get_current_milli_time
        self.get_worker_seqnum = p_get_worker_seqnum
        self.get_worker_ssrc = p_get_worker_ssrc
        self.mic_interface = 0
        self.connect_flag: bool = False
        self.bin_wrapper = BINWrapper()
        # self.device_index = 2

    def set_connect(self, p_connect_flag: bool):
        self.connect_flag = p_connect_flag
        print(f"EncodeMicPacketWorker connect:{self.connect_flag}")

    def run(self):
        while self.alive:
            while self.running:
                self.mic_interface = pyaudio.PyAudio()
                print(f"Mic Open, Mic Index:{self.device_index}")
                self.mic_stream = self.mic_interface.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True,
                                                          input_device_index=self.device_index,
                                                          frames_per_buffer=MIC_CHUNK)
                if self.mic_stream is None:
                    time.sleep(0.1)
                    # continue
                    break

                print(f"Mic End, Mic Index:{self.device_index} mic_stream:{self.mic_stream}")

                while self.running:
                    _frames = self.mic_stream.read(SPK_CHUNK, exception_on_overflow=False)

                    if _frames is None or self.join_flag is False:
                        time.sleep(0.1)
                        continue

                    audio_bin_data = self.bin_wrapper.to_bin_audio_data(_frames)
                    audio_bin_data = self.bin_wrapper.to_bin_wrap_common_header(timestamp=self.get_current_milli_time(),
                                                                                seqnum=self.get_worker_seqnum(),
                                                                                ssrc=self.get_worker_ssrc(),
                                                                                mediatype=TYPE_INDEX.TYPE_AUDIO,
                                                                                bindata=audio_bin_data)
                    self.send_grm_queue.put(audio_bin_data)

                    time.sleep(0.1)

                self.mic_stream.stop_stream()
                self.mic_stream.close()
                self.mic_interface.terminate()
                QApplication.processEvents()
                time.sleep(0.1)

            QApplication.processEvents()
            time.sleep(0.1)

        print("Stop EncodeMicPacketWorker")
        self.terminated = True
        # self.terminate()
