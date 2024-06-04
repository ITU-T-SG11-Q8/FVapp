import socket
import time
import threading
import numpy as np
import struct


class ThreadManager(threading.Thread):
    is_server = None
    bin_comm = None
    server_socket = None
    client_socket = None
    on_client_connected = None
    on_client_closed = None
    on_client_data = None

    def init_server(self, bin_comm, server_socket, client_connected, client_closed, client_data):
        self.is_server = True
        self.bin_comm = bin_comm
        self.server_socket = server_socket
        self.on_client_connected = client_connected
        self.on_client_closed = client_closed
        self.on_client_data = client_data
        print('init_server')

    def init_client(self, bin_comm, client_socket, client_connected, client_closed, client_data):
        self.is_server = False
        self.bin_comm = bin_comm
        self.client_socket = client_socket
        self.on_client_connected = client_connected
        self.on_client_closed = client_closed
        self.on_client_data = client_data
        print('init_client')

    data_len = -1
    data_all = np.array([])

    def run(self):
        if self.is_server is True:
            self.run_server()
        else:
            self.run_client()

    def run_server(self):
        while True:
            self.server_socket.listen()

            self.client_socket, addr = self.server_socket.accept()
            print(f'accept socket. {addr}')

            if self.on_client_connected is not None:
                self.on_client_connected()

            if self.bin_comm is not None:
                self.bin_comm.set_server_client_socket(self.client_socket)

            self.run_client()

            #  Do some 'work'
            if self.on_client_closed is not None:
                self.on_client_closed()

            if self.bin_comm is not None:
                self.bin_comm.set_server_client_socket(None)

            time.sleep(0.001)

    def run_client(self):
        recv_data = bytes()
        while True:
            try:
                # 데이터가 수신되면 클라이언트에 다시 전송합니다.(에코)
                data = self.client_socket.recv(2048)

                if not data:
                    print('>> Disconnected')
                    break

                recv_data = recv_data + data

                while True:
                    if len(recv_data) < 4:
                        break

                    pos = 0
                    #seq = struct.unpack("I", recv_data[pos:pos+4])[0]
                    #pos += 4

                    data_len = np.frombuffer(recv_data[pos:pos+4], dtype=np.int32)[0]
                    pos += 4
                    if len(recv_data) - pos < data_len:
                        break

                    _data = recv_data[pos:pos+data_len]
                    pos += data_len
                    recv_data = recv_data[pos:]

                    # print(f'############## data_len:{data_len} recv_len:{len(_data)}')
                    if self.on_client_data is not None:
                        self.on_client_data(_data)

                    time.sleep(0.001)

            except ConnectionResetError as e:
                print('>> Disconnected ' + str(e))
                break

            #  Do some 'work'
            time.sleep(0.001)


class BINComm:
    lock = None
    server_socket = None
    client_socket = None
    on_client_connected = None
    on_client_closed = None
    on_client_data = None
    seq = 0

    def __init__(self):
        self.lock = threading.Lock()

    def set_server_client_socket(self, client_socket):
        self.lock.acquire()
        self.client_socket = client_socket
        self.lock.release()

    def start_server(self, port, on_client_connected, on_client_closed, on_client_data):
        print(f'start comm server. port:{port}')

        self.on_client_connected = on_client_connected
        self.on_client_closed = on_client_closed
        self.on_client_data = on_client_data

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('', port))

        t = ThreadManager(name='server')
        t.init_server(self, self.server_socket, on_client_connected, on_client_closed, on_client_data)
        t.daemon = True

        t.start()

    def start_client(self, server, port, on_client_connected, on_client_closed, on_client_data):
        print(f'start comm client. server:{server} port:{port}')

        self.on_client_connected = on_client_connected
        self.on_client_closed = on_client_closed
        self.on_client_data = on_client_data

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((server, port))

        t = ThreadManager(name='client')
        t.init_client(self, self.client_socket, on_client_connected, on_client_closed, on_client_data)
        t.daemon = True
        t.start()

    def send_bin(self, bin_data):
        self.lock.acquire()
        if self.client_socket is not None:
            try:
                byte_data_len = len(bin_data)
                byte_data_len = np.array([byte_data_len])
                byte_data_len = byte_data_len.tobytes()

                # print(f'#### send. byte_data_len:{len(bin_data)} sent_len:{len(byte_data_len + bin_data)}')
                self.client_socket.send(byte_data_len + bin_data)

                time.sleep(0.001)
            except ConnectionResetError as e:
                self.client_socket = None
                if self.on_client_closed is not None:
                    self.on_client_closed()

                print('>> Disconnected ' + str(e))

        self.lock.release()

